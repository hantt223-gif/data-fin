import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# --- Khởi tạo Session State cho Lịch sử Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data_for_ai_markdown" not in st.session_state:
    st.session_state.data_for_ai_markdown = None

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # ******************************* PHẦN SỬA LỖI BẮT ĐẦU *******************************
    # Lỗi xảy ra khi dùng .replace() trên giá trị đơn lẻ (numpy.int64).
    # Sử dụng điều kiện ternary để xử lý giá trị 0 thủ công cho mẫu số.
    
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    # ******************************* PHẦN SỬA LỖI KẾT THÚC *******************************
    
    return df

# --- Hàm gọi API Gemini cho Phân tích Tổng quan ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- HÀM MỚI: Xử lý chat tương tác và duy trì lịch sử ---
def chat_with_gemini(data_for_ai, prompt, history, api_key):
    """Xử lý phiên chat tương tác với Gemini."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        # Thiết lập context ban đầu cho AI
        system_instruction = (
            "Bạn là một Trợ lý phân tích tài chính chuyên nghiệp. "
            "Người dùng đã tải lên dữ liệu Báo cáo Tài chính và đã được xử lý. "
            "Hãy trả lời các câu hỏi của người dùng dựa trên ngữ cảnh này. "
            "Dữ liệu phân tích chi tiết:\n\n"
            f"{data_for_ai}"
        )
        
        # Chuyển đổi lịch sử chat của Streamlit sang định dạng của Gemini API
        # Lưu ý: role 'assistant' trong st.session_state tương ứng với 'model' trong Gemini API
        gemini_history = []
        for m in history:
            role = 'model' if m['role'] == 'assistant' else 'user'
            gemini_history.append({"role": role, "parts": [{"text": m["content"]}]})
        
        # Thêm prompt hiện tại của người dùng
        contents = gemini_history + [{"role": "user", "parts": [{"text": prompt}]}]
        
        # Khởi tạo model và gửi yêu cầu, sử dụng system_instruction
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config={"system_instruction": system_instruction}
        )
        return response.text

    except APIError as e:
        return f"Lỗi Gemini: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# Khởi tạo giá trị mặc định cho chỉ số thanh toán
thanh_toan_hien_hanh_N = "N/A"
thanh_toan_hien_hanh_N_1 = "N/A"

# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            try:
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán, xử lý chia cho 0
                if no_ngan_han_N != 0:
                    thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                if no_ngan_han_N_1 != 0:
                    thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần" if isinstance(thanh_toan_hien_hanh_N_1, float) else str(thanh_toan_hien_hanh_N_1)
                    )
                with col2:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần" if isinstance(thanh_toan_hien_hanh_N, float) else str(thanh_toan_hien_hanh_N),
                        delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}" if isinstance(thanh_toan_hien_hanh_N, float) and isinstance(thanh_toan_hien_hanh_N_1, float) else None
                    )
                    
            except IndexError:
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
            except ZeroDivisionError:
                st.warning("Không thể tính Chỉ số Thanh toán Hiện hành do giá trị 'Nợ Ngắn Hạn' bằng 0.")


            # --- Chuẩn bị dữ liệu cho AI (Phải nằm ngoài khối button) ---
            try:
                tsnh_tang_truong = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]
            except IndexError:
                tsnh_tang_truong = "Không tìm thấy"
            
            data_for_ai_df = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{tsnh_tang_truong:.2f}%" if isinstance(tsnh_tang_truong, float) else str(tsnh_tang_truong),
                    f"{thanh_toan_hien_hanh_N_1:.2f}" if isinstance(thanh_toan_hien_hanh_N_1, float) else str(thanh_toan_hien_hanh_N_1), 
                    f"{thanh_toan_hien_hanh_N:.2f}" if isinstance(thanh_toan_hien_hanh_N, float) else str(thanh_toan_hien_hanh_N)
                ]
            })
            st.session_state.data_for_ai_markdown = data_for_ai_df.to_markdown(index=False)
            
            
            # --- Chức năng 5: Nhận xét AI Tổng quan ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI Tổng quan)")
            
            if st.button("Yêu cầu AI Phân tích"):
                api_key = st.secrets.get("GEMINI_API_KEY") 
                
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(st.session_state.data_for_ai_markdown, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
    st.session_state.data_for_ai_markdown = None # Đảm bảo data_for_ai là None khi chưa tải file

# --------------------------------------------------------------------------------
# --- CHỨC NĂNG MỚI: KHUNG CHAT HỎI ĐÁP TƯƠNG TÁC ---
# --------------------------------------------------------------------------------
st.divider()
st.header("6. Khung Chat Tương Tác (Hỏi đáp chuyên sâu với Gemini)")

# Kiểm tra điều kiện để bật khung chat
if st.session_state.data_for_ai_markdown is None:
    st.warning("Vui lòng tải lên Báo cáo Tài chính và xử lý thành công ở mục 1 để có thể sử dụng khung chat.")
else:
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets để dùng Chat.")
    else:
        # Hiển thị lịch sử tin nhắn
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Xử lý input từ người dùng
        if prompt := st.chat_input("Hỏi Gemini về báo cáo tài chính này (VD: 'Giải thích về tỷ trọng Nợ Ngắn Hạn')"):
            # Thêm tin nhắn của người dùng vào lịch sử
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Hiển thị tin nhắn của người dùng
            with st.chat_message("user"):
                st.markdown(prompt)

            # Lấy phản hồi từ Gemini
            with st.chat_message("assistant"):
                with st.spinner('Đang phân tích dữ liệu và phản hồi...'):
                    # Gọi hàm chat_with_gemini, truyền dữ liệu phân tích và lịch sử chat
                    full_response = chat_with_gemini(
                        st.session_state.data_for_ai_markdown, 
                        prompt, 
                        st.session_state.messages, 
                        api_key
                    )
                
                # Hiển thị và ghi lại phản hồi của AI
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
