# Use a pipeline as a high-level helper
from transformers import pipeline

classifier = pipeline("image-classification", model="trpakov/vit-face-expression")
predicted = classifier('https://storage.googleapis.com/kagglesdsdata/datasets/786787/1351797/test/neutral/PrivateTest_15847006.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240905%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240905T051634Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=a7245c8b140e55e779cf88eeb4377e8074ebaff2ac000bb05e9a5f77ee54bc8fefdfef49681c2f15a9a8949b4c8395342c7d04066c4bf0742ed0a03521d95279e3dffb2e0135bd1330fa0474ad7745719b4e4a8fce7ad1f9a921332e1d39bbc95efe280a962b39a23264836351d97bbc5e8c12e8352f16a7a469f60b48b0e3f591d9c0021452938e1effa5398eb5c8ed4b8cfca501dbd157eeec46adf33a3d60322a0fa81ad6b73e25c36f8131ccf20d7b1a3af3aa266784220c82a7e027ea3b80a68315ec9702aa093125e61641c7cb15daa825f79e1ef481b460cca907b4cc253e8a8264288de0a32e3df3a2b064bebcee14e6ba1d27ec6de91daf0ffe9872')
print(predicted)


######################################
            #CREDITS
######################################

# https://huggingface.co/google/vit-base-patch16-224-in21k
# https://huggingface.co/trpakov/vit-face-expression

# @misc{wu2020visual,
#       title={Visual Transformers: Token-based Image Representation and Processing for Computer Vision}, 
#       author={Bichen Wu and Chenfeng Xu and Xiaoliang Dai and Alvin Wan and Peizhao Zhang and Zhicheng Yan and Masayoshi Tomizuka and Joseph Gonzalez and Kurt Keutzer and Peter Vajda},
#       year={2020},
#       eprint={2006.03677},
#       archivePrefix={arXiv},
#       primaryClass={cs.CV}
# }

# @inproceedings{deng2009imagenet,
#   title={Imagenet: A large-scale hierarchical image database},
#   author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
#   booktitle={2009 IEEE conference on computer vision and pattern recognition},
#   pages={248--255},
#   year={2009},
#   organization={Ieee}
# }
