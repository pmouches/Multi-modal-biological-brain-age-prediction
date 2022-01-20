library(oro.nifti)
library(pTFCE)

# Output image directory 
imageDirectory=''
# Saliency map
Z=readNIfTI('.nii.gz')
# Brain mask
MASK=readNIfTI(".nii.gz")

# Performs ptfce
pTFCE=ptfce(Z, MASK)
newZ = pTFCE$Z

# Find the threshold for alpha=0.01
fwer0.01.Z=fwe.p2z(pTFCE$number_of_resels, 0.01)
# Threshold the image
super_threshold_indices = newZ < fwer0.01.Z
newZ[super_threshold_indices] = 0

# Write the thresholded image
writeNIfTI(newZ, paste(imageDirectory,"pTFCE_result.nii.gz", sep=""))
