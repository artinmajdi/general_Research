import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
sys.path.append('/array/ssd/msmajdi/code/general')
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from otherFuncs import smallFuncs
import uncrop
from uncrop import uncrop_by_mask
from nilearn import image as niImage
import nibabel as nib
# params.preprocess.Mode = False
# params.directories.Train.address
# subF = smallFuncs.listSubFolders(params.directories.Train.address)

dirr = '/array/ssd/msmajdi/experiments/keras/exp4/test/Main/vimp2_case3/' 
dir_ref    = '/array/ssd/msmajdi/experiments/keras/exp3/train/Main/vimp2_819_05172013_DS/'


ref = nib.load(dir_ref + 'WMnMPRAGE_bias_corr.nii.gz')
im = nib.load(dirr + 'LIFUP003_MPRAGE_WMn.nii.gz')

im2 = niImage.resample_img(img=im , target_affine=ref.affine  , target_shape=ref.get_data().shape , interpolation='continuous')
nib.save(im2, dirr + 'WMnMPRAGE_bias_corr.nii.gz')


dir_in = dirr + 'left/'
dir_out = smallFuncs.mkDir(dirr + 'Label/')
for label in smallFuncs.Nuclei_Class(method='Cascade').All_Nuclei().Names:

    print(label)
    uncrop_by_mask(input_image= dir_in + label + '.nii.gz', output_image=dir_out + label + '2.nii.gz' , full_mask=dir_in + 'mask_inp.nii.gz')

    ref = nib.load(dir_ref + 'Label/' + label + '.nii.gz')
    msk = nib.load(dir_out + label + '2.nii.gz')
    msk2 = niImage.resample_img(img=msk , target_affine=ref.affine  , target_shape=ref.get_data().shape , interpolation='nearest')
    nib.save(msk2, dir_out + label + '.nii.gz')
