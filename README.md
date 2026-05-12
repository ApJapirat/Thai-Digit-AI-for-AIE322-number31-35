# Thai Digit AI 31-35

โปรเจคนี้เป็นเว็บแอป Flask สำหรับเก็บตัวอย่างลายมือขนาด `28x28` และทำนายเลขภาษาไทยจาก `๓๑` ถึง `๓๕`.

โครงการนี้ออกแบบให้ใช้งานได้ง่ายบน PythonAnywhere free-tier โดยใช้เพียง Flask, scikit-learn, Pillow, NumPy และ joblib เท่านั้น ไม่มีการพึ่งพา TensorFlow หรือ Keras.

## คุณสมบัติหลัก

- วาดและบันทึกตัวอย่าง dataset สำหรับ label `31`, `32`, `33`, `34`, `35`
- ฝึกโมเดล scikit-learn ที่น้ำหนักเบา
- เก็บโมเดลที่ฝึกแล้วไว้ใน `model_versions/`
- เลือกโมเดลที่ใช้งานได้จากหน้า Admin
- ทำนายด้วยโมเดล `model.joblib`
- แสดง preview และ probability ของแต่ละ label บนหน้า Predict

## โครงสร้างโปรเจค

```text
.
├── create_weak_model.py
├── image_preprocessing.py
├── main.py
├── model_manager.py
├── README.md
├── requirements.txt
├── train.py
├── dataset/
│   ├── 31/
│   ├── 32/
│   ├── 33/
│   ├── 34/
│   ├── 35/
│   └── metadata.csv
├── demo_models/
├── model_versions/
├── static/
│   └── style.css
└── templates/
    ├── admin.html
    ├── index.html
    └── predict.html
```

## เส้นทางหลักของเว็บ

- `GET /` - หน้าเขียนตัวเลขและบันทึกตัวอย่างลง dataset
- `GET /predict-page` - หน้าวาดแล้วทดสอบการทำนาย
- `GET /admin` - เข้าหน้าแอดมิน, อัปโหลดโมเดล, ดูเมตริกซ์, สลับเวอร์ชัน
- `POST /predict` - ส่งภาพ base64 เพื่อทำนาย
- `POST /upload-model` - อัปโหลดและตรวจสอบโมเดล `.joblib`
- `POST /activate-model-version` - เปิดใช้งานโมเดลที่เก็บใน `model_versions/`
- `GET /download-dataset` - ดาวน์โหลด dataset เป็น ZIP

## ข้อตกลงโมเดล

- โมเดลที่ใช้งานอยู่: `model.joblib`
- ฟอร์แมตโมเดล: scikit-learn estimator เก็บด้วย joblib
- รูปร่างข้อมูลเข้า: `(1, 784)`
- ข้อมูลเข้า: แปลงจาก preprocessing ให้เป็นฟีเจอร์ 784 ค่า
- คลาส: `31`, `32`, `33`, `34`, `35`
- ป้ายผลลัพธ์ภาษาไทย: `๓๑`, `๓๒`, `๓๓`, `๓๔`, `๓๕`

ระบบจะทำงานได้ดีที่สุดกับ classifier ที่มี `predict_proba` ถ้าโมเดลมีแค่ `predict` จะให้ความมั่นใจ `confidence` = `1.0` แก่คลาสที่ทำนายได้

## การเตรียมภาพก่อนส่งโมเดล

ทั้งการฝึกและการทำนายผ่านเว็บใช้ [image_preprocessing.py](image_preprocessing.py) ร่วมกัน

ขั้นตอน preprocessing คือ:

- วางภาพบนพื้นหลังสีขาว
- แปลงภาพเป็นระดับสีเทา
- ย่อขนาดเป็น `28x28`
- ตรวจสอบและกลับสีอัตโนมัติเมื่อพื้นหลังมืด
- ตัดขอบภาพรอบหมึกเขียน
- ย่อหรือขยายตัวเลขให้พอดีในกรอบ `20x20`
- วางภาพตัวเลขตรงกลางบน canvas ขาว `28x28`
- ปรับค่า pixel ให้เป็นช่วง `0..1`
- แปลงเป็นเวกเตอร์ยาว `784`

การทำ preprocessing ให้เหมือนกันทั้งฝั่งเว็บและ dataset สำคัญมาก เพื่อให้โมเดลเรียนรู้และทำนายได้ตรงตามรูปแบบ

## ติดตั้ง

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

โฟลเดอร์ dataset ต้องมีโครงสร้างดังนี้:

```text
dataset/
  31/
  32/
  33/
  34/
  35/
  metadata.csv
```

แต่ละโฟลเดอร์ต้องมีไฟล์ PNG ของป้ายเลขนั้นๆ

## ฝึกโมเดล

รันคำสั่ง:

```powershell
python train.py
```

`train.py` จะฝึกโมเดลต่อไปนี้:

- KNN
- Decision Tree
- Random Forest
- SVM RBF

โมเดลที่ฝึกแล้วจะถูกบันทึกไว้ใน:

```text
model_versions/knn.joblib
model_versions/decision_tree.joblib
model_versions/random_forest.joblib
model_versions/svm_rbf.joblib
```

โมเดลที่ได้คะแนน validation สูงสุดจะถูกคัดลอกไปยัง:

```text
model.joblib
```

นอกจากนี้ยังสร้างไฟล์:

- `training_metrics.json` - เมตริกซ์และโมเดลที่เลือก
- `model_info.json` - ข้อมูลเมตาเกี่ยวกับโมเดลที่ใช้งาน
- `confusion_matrix.png` - confusion matrix ของโมเดลที่ดีที่สุด

เมตริกซ์รวมถึงความแม่นยำ (accuracy), precision, recall, f1-score, confusion matrix, ชื่อไฟล์โมเดล และเวลาที่ฝึก

## หน้า Admin

เปิดหน้า:

```text
http://127.0.0.1:5000/admin
```

เข้าสาธิต:

```text
username: admin
password: 911
```

หน้า Admin สามารถ:

- อัปโหลดโมเดล `.joblib`
- ตรวจสอบความเข้ากันได้ของโมเดล
- แสดงรายการเวอร์ชันใน `model_versions/`
- แสดง accuracy, precision, recall, f1-score ของแต่ละโมเดล
- สลับไปใช้โมเดลที่ต้องการ

เมื่อกด activate ระบบจะตรวจสอบโมเดลก่อนแทนที่ `model.joblib` หากตรวจสอบไม่ผ่าน โมเดลเดิมจะยังคงทำงานอยู่

## ตรวจสอบการทำนาย

เปิดหน้า:

```text
http://127.0.0.1:5000/predict-page
```

หลังทำนายจะแสดง:

- ป้ายภาษาไทยที่ทำนายได้
- ความมั่นใจ
- รูปภาพ `28x28` ที่ส่งเข้าโมเดล
- ความน่าจะเป็นของแต่ละ label `31`, `32`, `33`, `34`, `35`

ผล JSON จาก `/predict` จะมี:

- `prediction`
- `label`
- `confidence`
- `processed_image`
- `probabilities`

## รันในเครื่อง

```powershell
python main.py
```

เปิดใช้งาน:

```text
http://127.0.0.1:5000/
http://127.0.0.1:5000/predict-page
http://127.0.0.1:5000/admin
```

## หมายเหตุ PythonAnywhere

ติดตั้ง dependencies ด้วย:

```bash
pip install -r requirements.txt
```

ไม่ต้องลงแพ็กเกจ deep-learning ใหญ่ๆ โปรเจคนี้ใช้โมเดลจาก scikit-learn และเก็บโมเดลที่สลับได้ใน `model_versions/`
