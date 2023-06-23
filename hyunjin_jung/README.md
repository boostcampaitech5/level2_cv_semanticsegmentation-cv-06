# MMSegmentation 실행 방법

MMsegmentation의 deeplabv3plus를 29 class의 multi-label로 학습시키기.

1. mmsegmentation 설치


1. config 파일 수정

   `mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-160k_ade20k-512x512.py`를 `deeplabv3plus_r50-d8_4xb4-160k_ade20k-512x512.py`로 수정


1. models 수정
   - input : (512,512)
   - outpu : (1024,1024)

   `mmsegmentation/mmseg/models/decode_heads/seg_aspp_head.py`를 `sep_aspp_head.py`로 수정


1. loss 등록
   - `mmsegmentation/mmseg/models/losses`에 `bce_dice_loss.py` 추가
   - `mmsegmentation/mmseg/models/losses/__init__.py`에 ```from .bce_dice_loss import BCEDiceLoss``` 추가
   - `mmsegmentation/mmseg/models/losses/__init__.py`의 `__all__`에 `'BCEDiceLoss'` 추가

1. `Train mmsegmentation.ipynb` 실행

<br>

* 수정 및 추가해야 하는 파일은 repository의 mmseg 폴더 참고.
<br>

---

# Unet 학습
1. smp 설치
1. `Unet.ipynb` 실행

---
# Ensemble

* hard voting 기반의 앙상블

