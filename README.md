# AnatomyArchive
A versatile package for research-purpose medical image analysis (currently for CT only) and visualization
## License

This code is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

You are free to use, modify, and share the code for **non-commercial research and educational purposes** with attribution to this source. 

⚠️ **Disclaimer**: This software is **not intended for clinical or commercial use**. Use at your own risk. The authors provide no warranty and accept no liability.

As long as you use functions defined or derived from this package for your research work, you agree to cite our work: 
Lei Xu*, Torkel B Brismar. Software architecture and manual for novel versatile CT image analysis toolbox -- AnatomyArchive. 
https://doi.org/10.48550/arXiv.2507.13901.

You are recommended to read the preprint to get more information about the package including how to use it for your own work.  

In case you've identify any errors when using the codes, you are welcome to report them by creating issues. The owner of the codes is happy to provide assistance within 
an appropriate time frame.

This package is based on the full body segmentation open-weight model:
Wasserthal, J., Breit, H.-C., Meyer, M.T., Pradella, M., Hinck, D., Sauter, A.W., Heye, T., Boll, D., Cyriac, J., Yang, S., Bach, M., Segeroth, M., 2023. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiology: Artificial Intelligence. https://doi.org/10.1148/ryai.230024

# Example for doing body composition
from totalsegmentator.config import setup_nnunet
from featureAnalyzer import body_component_analysis, NestedDict


if __name__ == '__main__':
  setup_nnunet()
  dir_data = DIR_DATA # Replace it by wherever you store the NifTI images. 
  #Though TotalSegmentator supports conversion of DICOM files to NifTi images, it won't work with complicated DIOM folder structures. Check dicomParser for more info.
  result_dict = NestedDict()
  target_config ={'refClassMap': 'total', # can be skipped if default 'total' is used.
                  'total': {'refObjUB': 'vertebrae_L1',
                            'refObjLB': 'pelvic',
                            'excludeProsthesisSamples': True,
                            'selectedObjs': ['liver'] # Add whatever should be included
                            },
                  'tissue_types':{'selectedObjs': ['subcutaneous_fat', 'torso_fat','skeletal_muscle'],
                                 'enforceMuscleRange': False
                                 }
                  }
  body_component_analysis(dir_data, result_dict, target_config)
  
  
