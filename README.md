# 봄여름가을겨울🌷
ResNet50(ImageNet pretrained)을 사용한 퍼스널 컬러 웜톤, 쿨톤 이진분류

## 프로젝트 개요

### 프로젝트 기간

2023.06.30 ~ 2023.08.11
### 프로젝트 설명 및 소개

현대사회에서 이미지의 생산 및 전달은 중요한 커뮤니케이션 활동으로 인식된다. 이미지 형성에 있어서 의복을 포함한 신체적 외모는 비언어적 단어로서 지각자의 즉각적인 판단을 이끌어 내어 한 사람의 인상과 사회적 가치를 평가하는데 중요한 역할을 한다.  

퍼스널 컬러는 사람의 얼굴에 가장 어울리는 색상을 찾는 미용이론으로, 색에 따라 본인이 가진 고유한 피부가 가장 건강해보이고, 그에 따라 이목구비의 입체감이 자연스럽게 살아나는 지의 여부에 따라 결정된다. 이러한 퍼스널 컬러가 개인의 이미지를 나타내는 데 중요한 역할을 하고있다.

본 프로젝트는 퍼스널 컬러를 과학적 근거로 정확히 진단하여 사용자에게 가장 어울리는 색을 추천한다. 또한 사용자와 피부색이 가장 비슷한 유명인과 그에 맞는 패션, 메이크업 등을 추천하는 서비스를 제공한다.
 

## 팀소개

### 팀원 구성 및 역할

![08](https://github.com/https-github-com-jiyezzang/four_seasons/assets/70656219/6fac5e9e-9000-44c1-8cf4-5f27c65b0d64)






## 프로젝트 진행 과정
![01](https://github.com/https-github-com-jiyezzang/four_seasons/assets/70656219/44a7dd8a-f62a-450a-b8c3-155ee643aba3)


## 요구사항(Requirements)

### 프로젝트 환경 및 버전
```python
import pkg_resources
import sys

def get_version(package):
    return pkg_resources.get_distribution(package).version

print("Python 버전:", sys.version)
print("Google Colab 버전:", get_version('google-colab'))
print("Numpy 버전:", np.__version__)
print("Matplotlib 버전:", matplotlib.__version__)
print("OpenCV 버전:", cv2.__version__)
print("Pillow 버전:", PIL.__version__)
```
```python
Python 버전: 3.10.12 (main, Jun  7 2023, 12:45:35) [GCC 9.4.0]
Google Colab 버전: 1.0.0
Numpy 버전: 1.22.4
Matplotlib 버전: 3.7.1
OpenCV 버전: 4.7.0
Pillow 버전: 8.4.0
```




## 레퍼런스

**데이터셋**

AI허브_가족 관계가 알려진 얼굴 이미지 데이터

https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=528

AI허브_한국인 안면 이미지

https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=83

**논문 및 프로젝트**

GitHub_Skin lesion detection from dermoscopic images using Convolutional Neural Networks

https://github.com/adriaromero/Skin_Lesion_Detection_Deep_Learning

고려대학교 세종캠퍼스_너는무슨톤팀_퍼스널컬러 분류기 프로젝트

https://www.youtube.com/watch?v=GOp4MVE3BnE

하영호·김대철·이철희·최명희 (2012), 색 공간 기반의 피부색 검출 방법과 관심 영역을 이용한 피부색, 한국화상학회

https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART001735941

이연진 (2012), 퍼스널 컬러 유형 분류를 위한 정량적 측정과 평가, 충남대학교 대학원 박사학위논문.

[http://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=8616828902117cb3ffe0bdc3ef48d419&keyword=퍼스널컬러 분류](http://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=8616828902117cb3ffe0bdc3ef48d419&keyword=%ED%8D%BC%EC%8A%A4%EB%84%90%EC%BB%AC%EB%9F%AC%20%EB%B6%84%EB%A5%98)

김용현·오유석·이정훈 (2018), 퍼스널 컬러 스킨 톤 유형 분류의 정량적 평가 모델 구축에 대한 연구, 한국의류학회지

https://scienceon.kisti.re.kr/commons/util/originalView.do?cn=JAKO201810237886055&oCn=JAKO201810237886055&dbt=JAKO&journal=NJOU00290617
