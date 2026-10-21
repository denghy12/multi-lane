"""Label-blind same-image donor assignment for a query-only intervention."""
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .paired_transforms import PairedFullPersonTransform


def eval_geometry(bbox, image_size, resize=256, crop_size=224):
    width, height = image_size
    box = PairedFullPersonTransform.target_box(bbox, width, height)
    if width <= height:
        rw, rh = resize, int(resize * height / width)
    else:
        rw, rh = int(resize * width / height), resize
    scaled = None if box is None else (box[0]*rw/width, box[1]*rh/height,
                                      box[2]*rw/width, box[3]*rh/height)
    crop = (round((rh-crop_size)/2), round((rw-crop_size)/2), crop_size, crop_size)
    geometry = PairedFullPersonTransform.geometry(scaled, crop, False)
    center_visible = False if scaled is None else (
        crop[1] <= (scaled[0]+scaled[2])/2 <= crop[1]+crop_size and
        crop[0] <= (scaled[1]+scaled[3])/2 <= crop[0]+crop_size)
    return geometry, center_visible


def make_pairs(sample_ids, boxes, sizes, minimum_visibility=.9):
    if not 0 < minimum_visibility <= 1 or not len(sample_ids)==len(boxes)==len(sizes):
        raise ValueError('Invalid pairing inputs')
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError('Duplicate sample IDs')
    groups=defaultdict(list); rows=[]
    for i,(sid,box,size) in enumerate(zip(sample_ids,boxes,sizes)):
        geometry,center=eval_geometry(box,size)
        visible=bool(geometry[5] and geometry[4]+1e-7>=minimum_visibility and center)
        rows.append(dict(sample_id=str(sid), donor_sample_id=str(sid), donor_index=i,
                         geometry=geometry.tolist(), center_visible=center,
                         sufficiently_visible=visible, eligible=False))
        if visible: groups[str(sid).rsplit('#person=',1)[0]].append(i)
    image_count=0
    for group in groups.values():
        # Exact duplicate boxes do not establish a second target person.
        unique={}
        for i in sorted(group,key=lambda j: str(sample_ids[j])):
            unique.setdefault(tuple(np.asarray(boxes[i],dtype=float).ravel()[:4]),i)
        group=list(unique.values())
        if len(group)<2: continue
        image_count+=1
        for position,i in enumerate(group):
            donor=group[(position+1)%len(group)]
            rows[i].update(donor_index=donor,donor_sample_id=str(sample_ids[donor]),eligible=True)
    return dict(minimum_visibility=minimum_visibility, center_required=True,
                pairing='lexicographic_sample_id_cycle_distinct_body_boxes',
                label_or_prediction_selection=False, people=len(rows),
                eligible_people=sum(r['eligible'] for r in rows),eligible_images=image_count,
                rows=rows)


def plan_for_source(source, minimum_visibility=.9):
    transform=source.multi_view_transform
    if transform.train or transform.full_crop_mode!='legacy':
        raise ValueError('Query pairs require original deterministic legacy evaluation crops')
    if transform.full_eval.transforms[0].size!=256:
        raise ValueError('Unexpected Full resize')
    sizes={}
    for path in set(source.file_paths):
        with Image.open(path) as im: sizes[path]=im.size
    return make_pairs(source.sample_ids, source.body_bboxes,
                      [sizes[p] for p in source.file_paths], minimum_visibility)


class AttributionDataset(Dataset):
    def __init__(self, source, plan, class_indices, swap=False):
        self.source,self.plan,self.class_indices,self.swap=source,plan,list(class_indices),swap

    def __len__(self): return len(self.source)

    def __getitem__(self,index):
        images,target=self.source[index]
        row=self.plan['rows'][index]
        # Check the analytical crop geometry against the actual pixel transform.
        if not torch.allclose(images['bbox'],torch.tensor(row['geometry']),atol=1e-6,rtol=0):
            raise RuntimeError('Query eligibility geometry disagrees with actual Full crop')
        if self.swap:
            donor=images['person']
            if row['eligible']:
                j=row['donor_index']
                with Image.open(self.source.file_paths[j]) as im:
                    crop=self.source.crop_person(im.convert('RGB'),self.source.body_bboxes[j],
                                                 self.source.multi_view_transform.margin)
                    donor=self.source.multi_view_transform.person_eval(crop)
            images=dict(images,cross_view_query_donor=donor,
                        cross_view_query_mask=torch.tensor(row['eligible'],dtype=torch.bool))
        return images,target[self.class_indices],self.source.sample_ids[index],index
