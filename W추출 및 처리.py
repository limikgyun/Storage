from tensorflow.keras.models import load_model
import numpy as np

def process_flattened_weights(flattened_base_weights, flattened_transfer_weights_list):
    # base_np 행렬을 flattened_base_weights와 동일한 shape로 0으로 초기화
    base_np = np.zeros_like(flattened_base_weights)
    
    for idx in range(len(flattened_base_weights)):
        W_0 = flattened_base_weights[idx]
        W_1, W_2, W_3 = [transfer_weights[idx] for transfer_weights in flattened_transfer_weights_list]
        
        if min(W_1, W_2, W_3) < W_0 and max(W_1, W_2, W_3) > W_0:
            base_np[idx] = 0
        else:
            base_np[idx] = np.mean([W_1 - W_0, W_2 - W_0, W_3 - W_0])
    
    return base_np

def save_weights_as_numpy(weights, save_path):
    np.save(save_path, weights)

def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in weights])

def reshape_weights(flattened_weights, layer_shapes):
    reshaped_weights = []
    start = 0
    for shape in layer_shapes:
        size = np.prod(shape)
        reshaped_weights.append(flattened_weights[start:start + size].reshape(shape))
        start += size
    return reshaped_weights

# 모델 경로 리스트 (전이학습 전과 후의 모델 경로를 각각 나열)
base_model_path = '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/y/240905-1425-GAN-g-3840samples-100.h5'
transfer_model_paths = [
'/home/mnetlig/Desktop/CSI-SemiGAN-master/models/t-240920-1953-g-640샘플-60에폭-26.h5',
'/home/mnetlig/Desktop/CSI-SemiGAN-master/models/t-240920-1954-g-640샘플-60에폭-26.h5',
'/home/mnetlig/Desktop/CSI-SemiGAN-master/models/t-240920-1955-g-640샘플-60에폭-27.h5',
]

# 1단계: 다수의 모델의 weights를 불러오기
base_model = load_model(base_model_path)
base_weights = base_model.get_weights()

transfer_models = [load_model(path) for path in transfer_model_paths]
transfer_weights_list = [model.get_weights() for model in transfer_models]

# 2단계: 가중치를 일렬로 펼치기
flattened_base_weights = flatten_weights(base_weights)
flattened_transfer_weights_list = [flatten_weights(weights) for weights in transfer_weights_list]

# 3단계: 일렬로 펼친 가중치를 process_flattened_weights 함수로 처리
processed_flattened_weights = process_flattened_weights(flattened_base_weights, flattened_transfer_weights_list)

# 4단계: 처리된 가중치를 원래 형태로 복원
layer_shapes = [
    (100, 7808),
    (7808,),
    (1, 5, 32, 32),
    (32,),
    (1, 5, 32, 32),
    (32,),
    (1, 5, 32, 32),
    (32,),
    (1, 5, 1, 32),
    (1,)
]
reshaped_processed_weights = reshape_weights(processed_flattened_weights, layer_shapes)



# # 5단계: 결과를 저장
# save_weights_as_numpy(reshaped_processed_weights, 'processed_weights.npy')

# 6단계: 새로운 모델에 processed_weights를 더하기
new_model = load_model(base_model_path)  # 새로운 모델을 로드하거나 생성
new_model_weights = new_model.get_weights()

# 기존 가중치에 processed_weights를 더하기
updated_weights = [new_layer + processed_layer for new_layer, processed_layer in zip(new_model_weights, reshaped_processed_weights)]

# 새로운 가중치 설정
new_model.set_weights(updated_weights)

# 새로운 모델을 저장 (선택 사항)
new_model.save('/home/mnetlig/Desktop/CSI-SemiGAN-master/models/new_weighted_g.h5')
print("새 모델 저장 성공")