import cv2
import pytesseract
import os

# Đường dẫn tới tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Thiết lập biến môi trường TESSDATA_PREFIX
os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdata"

# Khởi tạo camera để đọc hình ảnh từ webcam
cap = cv2.VideoCapture(1)
# Vòng lặp xử lý các khung hình từ webcam
while (True):
    # Đọc một khung hình từ webcam
    ret, frame = cap.read()
    # Chuyển đổi khung hình sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Áp dụng ngưỡng thích nghi để phân đoạn ảnh
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # Hiển thị văn bản "KHUNG BIEN SO" trên khung hình
    cv2.putText(frame, "KHUNG BIEN SO", (40, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
    # Tìm các đường viền trong ảnh
    contours, h = cv2.findContours(thresh, 1, 2)
    largest_rectangle = [0, 0]
    for cnt in contours:
        lenght = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, lenght, True)
        # Kiểm tra xem đường viền có phải là một hình chữ nhật (có 4 điểm góc)
        if len(approx) == 4:
            area = cv2.contourArea(cnt)
            # Chọn đường viền có diện tích lớn nhất
            if area > largest_rectangle[0]:
                largest_rectangle = [cv2.contourArea(cnt), cnt, approx]
    # Lấy tọa độ của hình chữ nhật lớn nhất
    x, y, w, h = cv2.boundingRect(largest_rectangle[1])
    # Cắt phần hình ảnh chứa biển số xe
    image = frame[y:y + h, x:x + w]
    # Vẽ đường viền xung quanh biển số xe trên khung hình gốc
    cv2.drawContours(frame, [largest_rectangle[1]], 0, (0, 255, 0), 2)
    # Hiển thị văn bản "BIEN SO" trên khung hình
    cv2.putText(frame, "BIEN SO", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255))
    # Cửa sổ hiển thị camera
    cv2.imshow('Dinh Vi Bien So Xe', frame)

    # Chuyển đổi phần hình ảnh chứa biển số xe sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Áp dụng bộ lọc Gaussian để làm mờ ảnh
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # Áp dụng ngưỡng nhị phân với Otsu's binarization
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Cửa sổ hiển thị ảnh biển số sau khi áp dụng ngưỡng nhị phân
    cv2.imshow('Bien So La', thresh)

    # Áp dụng phép biến đổi hình thái học (morphological opening) để loại bỏ nhiễu
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    # Đảo ngược ảnh để văn bản trở thành màu đen trên nền trắng
    invert = 255 - opening
    # Sử dụng Tesseract OCR để nhận diện văn bản từ ảnh biển số xe
    data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
    # In kết quả nhận diện biển số xe ra màn hình
    print("Bien so xe la:")
    print(data)

    # Nhấn phím ESC để thoát
    key = cv2.waitKey(1)
    if key == 27:
        break

# Giải phóng tài nguyên và đóng tất cả các cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()
