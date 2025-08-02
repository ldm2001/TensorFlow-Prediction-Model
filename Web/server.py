# server.py
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import joblib
import os

app = Flask(__name__)

# 전역 변수로 모델 관련 객체 저장
W = None
b = None
scaler_X = None
scaler_y = None
model_loaded = False

# 모델 및 스케일러를 로드하는 함수
def load_model_and_scalers():
    global W, b, scaler_X, scaler_y, model_loaded
    
    try:
        print("모델 로딩 시작")
        
        # 1. 가중치와 편향 초기화
        W = tf.Variable(tf.random.normal([4, 1], dtype=tf.float64), name="weight")
        b = tf.Variable(tf.random.normal([1], dtype=tf.float64), name="bias")
        
        # 2. 체크포인트 파일 찾기
        checkpoint_dir = './model'
        
        # 최신 체크포인트 찾기
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        
        if checkpoint:
            print(f"체크포인트 발견: {checkpoint}")
            # 체크포인트 로드
            ckpt = tf.train.Checkpoint(W=W, b=b)
            ckpt.restore(checkpoint).expect_partial()
        else:
            # 구버전
            checkpoint_path = os.path.join(checkpoint_dir, 'saved.ckpt-1')
            if os.path.exists(checkpoint_path + '.index'):
                print(f"구버전 체크포인트 사용: {checkpoint_path}")
                ckpt = tf.train.Checkpoint(W=W, b=b)
                ckpt.restore(checkpoint_path).expect_partial()
            else:
                raise FileNotFoundError("체크포인트 파일을 찾을 수 없습니다.")
        
        # 3. 스케일러 로드
        scaler_X_path = os.path.join(checkpoint_dir, 'scaler_X.pkl')
        scaler_y_path = os.path.join(checkpoint_dir, 'scaler_y.pkl')
        
        if os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path):
            print("저장된 스케일러를 로드합니다")
            scaler_X = joblib.load(scaler_X_path)
            scaler_y = joblib.load(scaler_y_path)
        else:
            print("스케일러 파일이 없습니다 데이터에서 새로 생성합니다")
            # 학습 데이터에서 스케일러 생성
            import pandas as pd
            from sklearn.preprocessing import StandardScaler
            
            data = pd.read_csv('price data.csv')
            X = data[['avgTemp', 'minTemp', 'maxTemp', 'rainFall']].values
            y = data[['avgPrice']].values
            
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            scaler_X.fit(X)
            scaler_y.fit(y)
        
        model_loaded = True
        print("모델과 스케일러 로드 완료")
        
        # 로드된 모델 정보 출력
        print(f"가중치 형태: {W.shape}")
        print(f"편향 형태: {b.shape}")
        
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        model_loaded = False

# 가설 함수
def hypothesis(X):
    return tf.matmul(X, W) + b

# 애플리케이션 시작 시 모델 로드
load_model_and_scalers()

@app.route('/', methods=['GET', 'POST'])
def predict():
    price = None
    error_message = None
    
    if request.method == 'POST':
        if not model_loaded:
            error_message = "모델이 로드되지 않았습니다 먼저 모델을 학습시켜주세요"
        else:
            try:
                # 입력값 받기
                avg_temp = float(request.form.get('avg_temp', 0))
                min_temp = float(request.form.get('min_temp', 0))
                max_temp = float(request.form.get('max_temp', 0))
                rain_fall = float(request.form.get('rain_fall', 0))
                
                # 입력값 검증
                if min_temp > max_temp:
                    error_message = "최저 기온은 최고 기온보다 높을 수 없습니다"
                elif avg_temp < min_temp or avg_temp > max_temp:
                    error_message = "평균 기온은 최저와 최고 기온 사이여야 합니다"
                elif rain_fall < 0:
                    error_message = "강수량은 음수일 수 없습니다"
                else:
                    # 입력 데이터를 배열로 변환
                    data = np.array([[avg_temp, min_temp, max_temp, rain_fall]], dtype=np.float64)
                    
                    # 정규화
                    data_scaled = scaler_X.transform(data)
                    
                    # 예측
                    prediction_scaled = hypothesis(data_scaled).numpy()
                    
                    # 역정규화 
                    prediction = scaler_y.inverse_transform(prediction_scaled)
                    price = int(prediction[0][0])
                    
                    print(f"입력: 평균 {avg_temp}°C, 최저 {min_temp}°C, 최고 {max_temp}°C, 강수량 {rain_fall}mm")
                    print(f"예측 가격: {price}원")

            except ValueError:
                error_message = "올바른 숫자를 입력해주세요"
            except Exception as e:
                print(f"Error occurred: {e}")
                error_message = "예측 중 오류가 발생했습니다"

    return render_template('index.html', price=price, error=error_message)

# API 엔드포인트
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API for predictions"""
    if not model_loaded:
        return {'error': '모델이 로드되지 않았습니다'}, 500
    
    try:
        data = request.get_json()
        
        avg_temp = float(data.get('avg_temp', 0))
        min_temp = float(data.get('min_temp', 0))
        max_temp = float(data.get('max_temp', 0))
        rain_fall = float(data.get('rain_fall', 0))
        
        # 입력값 검증
        if min_temp > max_temp:
            return {'error': '최저 기온은 최고 기온보다 높을 수 없습니다'}, 400
        elif avg_temp < min_temp or avg_temp > max_temp:
            return {'error': '평균 기온은 최저와 최고 기온 사이여야 합니다'}, 400
        elif rain_fall < 0:
            return {'error': '강수량은 음수일 수 없습니다'}, 400
        
        # 예측
        input_data = np.array([[avg_temp, min_temp, max_temp, rain_fall]], dtype=np.float64)
        data_scaled = scaler_X.transform(input_data)
        prediction_scaled = hypothesis(data_scaled).numpy()
        prediction = scaler_y.inverse_transform(prediction_scaled)
        price = int(prediction[0][0])
        
        return {
            'price': price,
            'input': {
                'avg_temp': avg_temp,
                'min_temp': min_temp,
                'max_temp': max_temp,
                'rain_fall': rain_fall
            }
        }
        
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/api/model_status')
def model_status():
    return {
        'model_loaded': model_loaded,
        'model_type': '선형 회귀',
        'input_features': ['avgTemp', 'minTemp', 'maxTemp', 'rainFall'],
        'output': 'avgPrice'
    }

if __name__ == "__main__":
    app.run(debug=True)
