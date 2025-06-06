import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import CNNLSTM_ImageRadFusion  # Kiến trúc mô hình mới
import os
import random
import numpy as np
import torch
from torchvision.models import resnet18
def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Set to False for determinism
    # torch.use_deterministic_algorithms(True) # For PyTorch 1.8+

# Gọi hàm set_seed với một số bất kỳ
set_seed(42) # Bạn có thể chọn bất kỳ số nguyên nào
# Dùng ResNet18 đã tiền huấn luyện để trích xuất đặc trưng
resnet = resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Bỏ lớp FC cuối cùng
resnet.eval()
# Khởi tạo mô hình
model = CNNLSTM_ImageRadFusion(input_dim=513, hidden_dim=128)

# Load trọng số
try:
    model.load_state_dict(torch.load("best_model_1.pth", map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    st.error(f"❌ Không thể load model: {e}")
    st.stop()

# Tạo projector từ 12288 → 512
projector = torch.nn.Linear(12288, 512)

# Hàm tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Giao diện Streamlit
st.title("📸 Dự báo bức xạ mặt trời từ ảnh")

uploaded_files = st.file_uploader("Tải lên chuỗi ảnh (theo thứ tự thời gian)", 
                                   type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.subheader("Ảnh đã tải lên:")
    cols = st.columns(min(5, len(uploaded_files)))
    for i, file in enumerate(uploaded_files):
        image = Image.open(file)
        cols[i % 5].image(image, width=100, caption=f"Ảnh {i+1}")

    if st.button("📈 Dự đoán từ ảnh"):
        try:
            tensors = []
            for file in uploaded_files:
                img = Image.open(file).convert("RGB")
                img_tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)
                with torch.no_grad():
                    feature = resnet(img_tensor)           # (1, 512, 1, 1)
                    feature = feature.view(-1)             # (512,)
                    tensors.append(feature)

            # Stack lại thành chuỗi (T, 512)
            image_features = torch.stack(tensors)  # (T, 512)

            # Thêm 1 đặc trưng bức xạ giả
            rad_dummy = torch.zeros(image_features.size(0), 1)
            combined_input = torch.cat((image_features, rad_dummy), dim=1)  # (T, 513)
            input_tensor = combined_input.unsqueeze(0)  # (1, T, 513)


            # Thêm batch dim → (1, T, 513)
            input_tensor = combined_input.unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                prediction = output.item()*10

            st.subheader("🌞 Dự báo bức xạ mặt trời:")
            st.success(f"🌤️ Giá trị bức xạ dự đoán: **{prediction:.2f} W/m²**")

        except Exception as e:
            st.error(f"❌ Lỗi dự đoán: {e}")
