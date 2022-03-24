##프로젝트 설명
KFood 음식 탐지 및 간단한 추천시스템
- 한식에 관심이 있는 외국인들에게 드라마,영화 속에서 발견한 한식 사진 또는 한국여행 중 찍은 한식 사진을 업로드하면 82종의 한식 중에서 어떤 한식인지 detection 및 classification해주고, 해당 음식에 대한 레시피, romanized name, 간단한 설명, 재료 등을 알려줍니다.

( 7팀 AI 프로젝트 최종발표.pdf를 참고하세요 )

또한 그 음식을 먹어보지 못했더라도 얼마나 본인의 입맛에 잘 맞을지 5개의 레벨로 알려주고, 매운맛, 신맛, 짠맛, 기름진 맛의 선호에 따라 잘 맞을만한 한식 상위 3개를 추천해줍니다.

( Kfood음식 82종 수작업_final.csv, cos_sim_추천시스템.ipynb를 참고하세요 )

이 레포지토리에는 AI파트의 업무 코드만 올렸기에 training과 inference, 파이썬 코드 실행(cos_sim_추천시스템.ipynb)을 통한 음식 추천만 가능합니다. 전체 웹서비스 코드는 추후 업데이트 하겠습니다.


##제작 상세 과정 블로그 및 노션

( 개발과정에서 겪었던 어려움들과 해결방법, 개발 일정들을 적은 블로그입니다. 한번 읽어주시면 감사하겠습니다 )




##사용방법

해결되지 않는 문제는 아래 yolov5 GitHub를 참고하세요.

https://github.com/ultralytics/yolov5


GitHub - ultralytics/yolov5: YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite

YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite. Contribute to ultralytics/yolov5 development by creating an account on GitHub.

github.com




1. 아래 링크에서 우리팀이 만든 데이터셋 다운받아서 zip 풀고 datasets/ 아래에 그림과 같이 폴더 설정하기



datasets/kfood 아래에 이렇게 파일 구성


![image](https://user-images.githubusercontent.com/74050826/159860819-83fc8205-68c6-4e64-8d9e-aa19c9840dbb.png)


![image](https://user-images.githubusercontent.com/74050826/159860887-7b0f1d92-2792-4171-9a52-6bd38861be36.png)


![image](https://user-images.githubusercontent.com/74050826/159860916-3f6abbe5-4e87-4217-8fca-15d390126289.png)


![image](https://user-images.githubusercontent.com/74050826/159860931-98a64697-e715-460e-b119-b3f88c906aed.png)



그리고 kfood.yaml에 datasets위치를 잘 설정해줍니다. 만약 음식탐지가 아닌 다른 데이터를 training시켜 탐지하고 싶다면 datasets 폴더 안에는 저희 kfood 데이터셋 뿐 아니라 coco 등 넣으시고 싶으신 데이터셋을 넣고 학습시킬 수도 있습니다. (마찬가지로 kfood.yaml 대신 coco.yaml 등 만들어서 사용하시면 됩니다)



2. 가상환경 만들기

```bash
# 가상환경 생성
python -m venv 가상환경이름

# 가상환경 activate - 이건 리눅스, Mac, 윈도우 등에 따라 다릅니다
source 가상환경이름/Scripts/activate 

# 가상환경에 requirements.txt 설치 - 필요한 라이브러리들 설치
pip install -r yolov5/requirements.txt
```


encoding 및 Upsampling 문제로 인해 저희 코드로 Yolov5를 돌리시려면 저희가 올린 requirements.txt로 pip install 하시기를 바랍니다. (Yolov5 GitHub의 requirements와 다릅니다)


![image](https://user-images.githubusercontent.com/74050826/159861149-680b98d5-6512-44fd-a0d8-fa1fb2f077b8.png)


![image](https://user-images.githubusercontent.com/74050826/159861167-2eb7d59e-24de-43b6-b332-f555fd830c7f.png)



여기까지 했다면 저희의 kfood 프로젝트를 이용할 준비가 완료되었습니다.



3. inference 하기

- team07_test.py 에서 input image, 사용할 모델, result를 저장할 경로 등 원하는 대로 수정해줍니다.

특히 opt.device = 'cpu'는 gpu가 없는 컴퓨터라면 꼭 넣기, gpu가 있다면 주석처리 하시기를 바랍니다.

![image](https://user-images.githubusercontent.com/74050826/159861198-10c38f9f-15cf-463e-9406-56419120814f.png)


그 다음 가상환경으로 들어간 후 아래 코드를 실행합니다.

```bash
python team07_test.py
```

![image](https://user-images.githubusercontent.com/74050826/159861239-039312d7-7b8c-41c9-af5c-5f5ddc234e14.png)


![image](https://user-images.githubusercontent.com/74050826/159861251-10a75889-8d0d-4c74-991e-bb3c9337d712.png)


위 사진과 같이 test_result 폴더 안에 음식이 detect된 결과 이미지가 저장된 것을 볼 수 있습니다.



4. train 하기

- pretrained된 모델 다운로드해오기 : yolov5dowonload.py 


![image](https://user-images.githubusercontent.com/74050826/159861311-b9ca36e8-b083-4de1-8ee2-d306c91bd7ba.png)


원하는 모델 종류를 골라서 수정하면 로컬에 .pt 확장자의 모델이 다운받아집니다.


![image](https://user-images.githubusercontent.com/74050826/159861324-0db1b0b7-6613-471e-9abc-062b9b1ece52.png)


- (권장)tmux 사용하기 : training은 매우 오랜 시간이 걸리는 과정입니다. 따라서 로컬에서 training을 시킨다면 컴퓨터를 장시간 켜놓는 것이 부담될 수 있습니다. tmux는 원격 GPU 서버를 사용하여 train 시킬 때 컴퓨터를 꺼도 돌아가게끔 해주는 session입니다. 개인 로컬 PC를 꺼놔도 원격으로 training을 진행할 수 있습니다.

우분투 또는 리눅스  tmux 설치:  sudo apt-get install tmux

windwos tmux 설치 : 아래 과정 참고

![image](https://user-images.githubusercontent.com/74050826/159861356-fa9f3d90-7125-44e0-bb34-14d45ebc47bb.png)


tmux 사용법
```bash
# 새로운 세션 생성
tmux new -s (session_name)

# 세션 목록
tmux ls

# 세션 다시 시작하기(다시 불러오기)
tmux attach -t session_number

# 세션 종료
exit
```

- 하이퍼파라미터 변경 : hyp.scratch-low.yaml 

![image](https://user-images.githubusercontent.com/74050826/159861422-0f224369-619c-41ec-a825-1f1aadc5fc0b.png)

- (권장) wandb 사용하기 : wandb는 머신러닝의 training과정을 시각화하고, logging 해주는 사이트입니다. 적절한 하이퍼파라미터를 찾기 위해 사용되기도 하고, training한 이력들을 확인하기 위해 쓰기도 합니다. loss 그래프, precision 그래프 등을 깔끔하게 볼 수 있습니다.

https://wandb.ai/site 회원가입 및 로그인

![image](https://user-images.githubusercontent.com/74050826/159861442-3b1ddbbd-81a0-461e-80ac-daff716bfb53.png)


- training run 시키기
```bash
(예시)
# Single GPU
python train.py --img 640 --batch 8 --epochs 10 --data kfood.yaml --weights yolov5x6.pt --name batch_8_0.002_epoch_10_v5x6

# Multi GPU
$ python -m torch.distributed.launch --nproc_per_node 2 train.py --batch 16 --epochs 10 --data kfood.yaml --weights yolov5x6.pt
```


--device 0,1 --name batch_16_0.002_epoch_10_v5x6
--batch 는 GPU 메모리에 따라 8의 배수로 결정합니다(byte 단위 때문).

--weights 에는 pretrained된 다운로드 받은 모델 종류를 적습니다.

--device 뒤에 오는 옵션은 cuda device를 말합니다. gpu 갯수에 따라 gpu 0 또는 gpu 0,1 gpu가 없으면 cpu 등 다양하게 올 수 있습니다.

--name 뒤에 오는 옵션은 train 시킬 모델의 이름입니다. 설정하신 하이퍼파라미터와 pretrained 모델 종류 등을 본따 지으시는 게 좋습니다. 옵션에 대한 설명들은 train.py 에 있습니다.

학습된 모델은 runs/train 안에 저장됩니다.
![image](https://user-images.githubusercontent.com/74050826/159861528-a2d2e22b-9c66-448c-9097-493e893d056e.png)


wandb 옵션 : yolov5 GitHub에서 wandb 관련 코드를 제공해주어서 train 시 1,2,3 중 3을 입력하면 wandb를 사용하지 않고 training합니다. 저는 2를 입력하여 사용해보겠습니다

![image](https://user-images.githubusercontent.com/74050826/159861542-a9f69a73-91f7-4cc0-ab4d-77f2f29f7e9d.png)


![image](https://user-images.githubusercontent.com/74050826/159861550-bbce226a-9597-4937-a6c6-c17e64ecd3b1.png)


![image](https://user-images.githubusercontent.com/74050826/159861562-9119a774-b10a-40bd-86bf-a8d88f7e4781.png)


![image](https://user-images.githubusercontent.com/74050826/159861601-d9346561-e299-49c0-a839-58eca7268fbc.png)


![image](https://user-images.githubusercontent.com/74050826/159861614-529c9d64-fce8-4a14-87e2-63d992dee242.png)


이렇게 training이 잘 실행되고 있고(지금은 로컬에서 CPU로 돌리는 상황이라 실제 training을 할 때는 주로 원격에서 GPU로 돌립니다.)

(권장 - wandb) 아래 사진처럼 wandb에도 잘 연결되는 것을 볼 수 있습니다. 아직은 1epoch이 지나지 않아 아무 그래프가 뜨지 않았습니다. 1 epoch이 지나고 나면 그래프가 뜨게 됩니다.


![image](https://user-images.githubusercontent.com/74050826/159861634-e88013fb-2bdf-4633-bc51-02e40eebc0c0.png)


![image](https://user-images.githubusercontent.com/74050826/159861660-8d384e0c-35eb-4795-8d37-bc3242d9a9d1.png)


![image](https://user-images.githubusercontent.com/74050826/159861726-bd448cd3-0839-43a4-ad40-c5f18a8b22c9.png)

