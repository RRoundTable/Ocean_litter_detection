# ocean_litter_detection
해양쓰레기 검출 프로젝트입니다.

모델의 용량이 커서 프로그램의 전부가 올라가지 않습니다.
학습된 모델을 이용하고 싶으시다면 아래의 링크를 클릭해주세요.

- faster_rcnn_inception_v2_coco : [inference_graph](https://1drv.ms/f/s!Aue5JNE50VRlh6MKUmXLta_mzNUSTA)
- faster_rcnn_resnet101_coco : [inference_graph_1](https://1drv.ms/f/s!Aue5JNE50VRlh6MNeyezEwzM6jPuUA)

## ocean_litter란

사람이 살면서 생긴 모든 부산물로써 바다로 들어가 못쓰게 된 것을 말합니다.

해양쓰레기는 근본적으로 육지의 쓰레기와 다르지 않습니다. 사람이 살면서 생긴 모든 부산물이 바다로 들어가 못쓰게 되면, 그것이 곧 해양쓰레기입니다. 육지에서 바다로 들어갔건, 바다에서 버려졌건 사람이 사용하는 모든 물건, 도구, 구조물 등이 해양쓰레기가 될 수 있습니다.

## 방법론

### data 수집

- 해양 오염에 대한 문제는 중대한 사안이나, 이에 관련된 이미지 데이터는 상대적으로 구하기 어려웠습니다.
- 따라서 구글에서 제공하는 API를 이용하여 데이터를 수집, 정제하였습니다.

[google_imag_download](https://github.com/hardikvasa/google-images-download)를 이용하여 'ocean_litter' image 다운로드를 하였습니다.

### bbox

- object detection에서 labeling은 model선정만큼 혹은 model보다 더 중요한 문제입니다.
- 다음 프로그램을 활용하여 lableling 데이터를 저장하였습니다.
[labelImg](https://github.com/tzutalin/labelImg)를 이용하여 bbox와 xml 데이터를 확보하였습니다.

### custom object detection(pretrained model)

object_detection은 복잡한 네트워크를 가지고 있기 때문에 처음부터 학습시키는 것은 다소 개인으로서는 힘들 수 있습니다.

따라서, 저는 pretrained된 모델을 사용하여 computation cost를 줄였습니다.

사용한 모델

- faster_rcnn_inception_v2_coco_2018_01_28(pretrained)
- faster_rcnn_resnet101_coco_2018_01_28((pretrained))

다양한 모델을 확인하고 싶다면 [model_zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)를 클릭하세요.


### evaluation


- loss

![loss](https://user-images.githubusercontent.com/27891090/50773648-b0cb7000-12d4-11e9-8cee-2d65975735e3.PNG)

computing power issue로 인해서 batch_size를 5로 고정하여 학습한 결과 학습이 부드럽게 진행되지 못하는 양상을 보이고 있으나,
전체적인 학습결과는 향상되는 모습을 보이고 있습니다.

- 실제 동영상에서 작동여부

![ezgif com-video-to-gif](https://user-images.githubusercontent.com/27891090/50773983-a1005b80-12d5-11e9-9bc2-01d491099204.gif)

### pdf 자료 

- 시연영상

![시연](./image/시연영상.gif)

- 실제환경에서 실험

![인하대 ](./image/인경호영상.gif)

- 기대효과

![기대효과](./image/기대효과.gif)

자세한 사항을 확인하고 싶다면 [pdf자료](https://github.com/RRoundTable/ocean_litter_detection/blob/master/pdf.pdf)를 확인해주세요.

pdf자료는 2019년 슈퍼챌린지 해커톤에서 발표한 자료입니다.

- 팀명 : 몽키스패너
- 딥러닝 개발 : 류원탁
- 기획 및 발표 : 구본우
- 서버 개발 : 어유선
- IOT 개발 : 이재혁, 

### reference 
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
