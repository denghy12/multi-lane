import unittest

import numpy as np
import torch

from scripts.emotic.d9_image_gate_readout import (fused, image_loss, image_rows, mask_means, permute_images,
                                                 probabilities)


class ImageReadoutTest(unittest.TestCase):
    def test_uniform_start_and_one_vector_for_all_people_and_heads(self):
        usable = torch.tensor([False, True])
        w = probabilities(torch.zeros(2,3),torch.zeros(1,3),torch.tensor([[1.],[3.]]),usable)
        torch.testing.assert_close(w,torch.tensor([[.5,.5,0],[1/3,1/3,1/3]]))
        inverse = torch.tensor([0,0,1])
        logits = torch.arange(3*8*3).float().reshape(3,8,3)
        prediction = fused(logits,w,inverse)
        torch.testing.assert_close(prediction[:2],logits[:2,:,:2].mean(dim=2))
        torch.testing.assert_close(prediction[2],logits[2].mean(dim=1))

    def test_image_loss_equalizes_images_of_different_sizes(self):
        logits = torch.tensor([[.5],[-1.]])
        labels = torch.tensor([[1.],[1.]])
        original = image_loss(logits,labels,torch.tensor([0,1]),torch.tensor([1,1]))
        repeated = torch.tensor([0,0,0,1])
        result = image_loss(logits[repeated],labels[repeated],repeated,torch.tensor([3,1]))
        torch.testing.assert_close(original,result)

    def test_exchange_preserves_availability_and_mean_by_image(self):
        ids = np.array(['c','b','a','d','e'])
        mask = np.array([True, False, False, True, True])
        rng = np.random.default_rng(42)
        donors = permute_images(ids,mask,rng).numpy()
        self.assertEqual(len(set(donors)),len(ids))
        np.testing.assert_array_equal(mask,mask[donors])
        self.assertFalse(np.any(donors == np.arange(len(ids))))
        w = torch.arange(15).reshape(5,3).float()
        result = mask_means(w,torch.from_numpy(mask))
        torch.testing.assert_close(result[0],w[mask].mean(dim=0))
        torch.testing.assert_close(result[2],w[~mask].mean(dim=0))

    def test_sample_keys_reject_split_leakage_and_duplicates(self):
        ids = np.array(['val:x#person=0','val:x#person=1'])
        np.testing.assert_array_equal(image_rows(ids,'val'),['val:x','val:x'])
        with self.assertRaises(ValueError):
            image_rows(ids,'train')
        with self.assertRaises(ValueError):
            image_rows(np.repeat(ids,2),'val')


if __name__ == '__main__':
    unittest.main()
