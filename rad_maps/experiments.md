# Radiomic feature maps statistical analysis results

## 24/01 
Run rad maps for features selected last run:
F1:  ['original_glrlm_GrayLevelNonUniformity']

F5: ['original_glrlm_GrayLevelNonUniformity', 'original_glszm_SizeZoneNonUniformityNormalized']

F1/F3: ['original_glcm_Contrast', 'original_glszm_LargeAreaHighGrayLevelEmphasis', 'original_ngtdm_Contrast']

F1/F4: ['original_glcm_JointEnergy', 'original_glcm_MaximumProbability', 'original_gldm_DependenceVariance']

F1/F5: ['original_firstorder_Entropy', 'original_gldm_DependenceEntropy', 'original_glrlm_GrayLevelNonUniformity']

## 05/02
Computed delta-rad entropy maps and clustered them: Patient 1 has no local relapse and show a decrease in entropy between F1 and F5 (shown by -1 value on clustered delta-rad map). 