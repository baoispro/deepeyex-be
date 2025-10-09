package emailservice

import (
	"fmt"
	"hospital-service/internal/config"

	"github.com/resend/resend-go/v2"
)

type EmailService struct {
	client *resend.Client
	cfg    config.Config
}

// NewEmailService khởi tạo email service mới
func NewEmailService(cfg config.Config) *EmailService {
	client := resend.NewClient(cfg.ResendAPIKey)
	return &EmailService{
		client: client,
		cfg:    cfg,
	}
}

// SendEmailRequest cấu trúc để gửi email
type SendEmailRequest struct {
	From    string   `json:"from" binding:"required"`
	To      []string `json:"to" binding:"required"`
	Subject string   `json:"subject" binding:"required"`
	HTML    string   `json:"html" binding:"required"`
	Text    string   `json:"text,omitempty"`
}

type OrderItem struct {
	ServiceID string     `json:"service_id"`
	ItemName  string  `json:"item_name"`
	Quantity  int     `json:"quantity"`
	Price     float64 `json:"price"`
}


// SendEmail gửi email tới khách hàng
func (s *EmailService) SendEmail(req SendEmailRequest) (string, error) {
	params := &resend.SendEmailRequest{
		From:    req.From,
		To:      req.To,
		Subject: req.Subject,
		Html:    req.HTML,
	}

	// Nếu có text version
	if req.Text != "" {
		params.Text = req.Text
	}

	sent, err := s.client.Emails.Send(params)
	if err != nil {
		return "", fmt.Errorf("failed to send email: %w", err)
	}

	return sent.Id, nil
}

// SendAppointmentConfirmation gửi email xác nhận lịch hẹn
func (s *EmailService) SendAppointmentConfirmation(
	toEmail, patientName, doctorName, appointmentDate, appointmentTime, appointmentCode string,
	orderItems []OrderItem, // thêm struct OrderItem để truyền vào
) error {

	// Tính tổng tiền
	var total float64
	for _, item := range orderItems {
		total += item.Price * float64(item.Quantity)
	}

	// Tạo danh sách sản phẩm HTML
	var orderDetailsHTML string
	for _, item := range orderItems {
		orderDetailsHTML += fmt.Sprintf(
			"<li>%s - %d x %s₫</li>",
			item.ItemName,
			item.Quantity,
			formatCurrency(item.Price),
		)
	}

	html := fmt.Sprintf(`
		<!DOCTYPE html>
		<html>
		<head>
			<meta charset="UTF-8">
			<style>
				body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
				.container { max-width: 600px; margin: 0 auto; padding: 20px; }
				.header { background-color: #4CAF50; color: white; padding: 20px; text-align: center; }
				.content { padding: 20px; background-color: #f9f9f9; }
				.footer { text-align: center; padding: 20px; font-size: 12px; color: #666; }
				.info-box { background-color: white; padding: 15px; margin: 10px 0; border-left: 4px solid #4CAF50; }
				ul { padding-left: 20px; }
			</style>
		</head>
		<body>
			<div class="container">
				<div class="header">
					<h1>Xác Nhận Lịch Hẹn</h1>
				</div>
				<div class="content">
					<p>Kính gửi <strong>%s</strong>,</p>
					<p>Lịch hẹn khám bệnh của bạn đã được xác nhận thành công!</p>
					
					<div class="info-box">
						<h3>Thông tin lịch hẹn:</h3>
						<p><strong>Mã đơn hàng:</strong> %s</p>
						<p><strong>Bác sĩ:</strong> %s</p>
						<p><strong>Ngày khám:</strong> %s</p>
						<p><strong>Giờ khám:</strong> %s</p>
						<p><strong>Tổng tiền:</strong> %s₫</p>
						
						<h4>Chi tiết sản phẩm:</h4>
						<ul>%s</ul>
					</div>
					
					<p>Vui lòng đến trước giờ hẹn 15 phút để làm thủ tục.</p>
					<p>Nếu có thay đổi, vui lòng liên hệ với chúng tôi sớm nhất có thể.</p>
					
					<p style="margin-top: 20px;">Trân trọng,<br><strong>DeepEyeX Medical Center</strong></p>
				</div>
				<div class="footer">
					<p>Email này được gửi tự động. Vui lòng không reply.</p>
					<p>&copy; 2025 DeepEyeX. All rights reserved.</p>
				</div>
			</div>
		</body>
		</html>
	`, patientName, appointmentCode, doctorName, appointmentDate, appointmentTime, formatCurrency(total), orderDetailsHTML)

	text := fmt.Sprintf(
		"Kính gửi %s,\n\nLịch hẹn của bạn đã được xác nhận!\n\nMã đơn hàng: %s\nBác sĩ: %s\nNgày: %s\nGiờ: %s\nTổng: %s₫\n\nTrân trọng,\nDeepEyeX Medical Center",
		patientName, appointmentCode, doctorName, appointmentDate, appointmentTime, formatCurrency(total),
	)

	req := SendEmailRequest{
		From:    "DeepEyeX <onboard@resend.dev>",
		To:      []string{toEmail},
		Subject: "Xác nhận lịch hẹn & đơn hàng",
		HTML:    html,
		Text:    text,
	}

	_, err := s.SendEmail(req)
	return err
}


// SendAppointmentReminder gửi email nhắc nhở lịch hẹn
func (s *EmailService) SendAppointmentReminder(toEmail, patientName, doctorName, appointmentDate, appointmentTime string) error {
	html := fmt.Sprintf(`
		<!DOCTYPE html>
		<html>
		<head>
			<meta charset="UTF-8">
			<style>
				body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
				.container { max-width: 600px; margin: 0 auto; padding: 20px; }
				.header { background-color: #FF9800; color: white; padding: 20px; text-align: center; }
				.content { padding: 20px; background-color: #f9f9f9; }
				.footer { text-align: center; padding: 20px; font-size: 12px; color: #666; }
				.info-box { background-color: white; padding: 15px; margin: 10px 0; border-left: 4px solid #FF9800; }
			</style>
		</head>
		<body>
			<div class="container">
				<div class="header">
					<h1>⏰ Nhắc Nhở Lịch Hẹn</h1>
				</div>
				<div class="content">
					<p>Kính gửi <strong>%s</strong>,</p>
					<p>Đây là email nhắc nhở về lịch hẹn khám bệnh của bạn.</p>
					
					<div class="info-box">
						<h3>Thông tin lịch hẹn:</h3>
						<p><strong>Bác sĩ:</strong> %s</p>
						<p><strong>Ngày khám:</strong> %s</p>
						<p><strong>Giờ khám:</strong> %s</p>
					</div>
					
					<p>Vui lòng đến trước giờ hẹn 15 phút để làm thủ tục.</p>
					
					<p style="margin-top: 20px;">Trân trọng,<br><strong>DeepEyeX Medical Center</strong></p>
				</div>
				<div class="footer">
					<p>Email này được gửi tự động. Vui lòng không reply.</p>
					<p>&copy; 2025 DeepEyeX. All rights reserved.</p>
				</div>
			</div>
		</body>
		</html>
	`, patientName, doctorName, appointmentDate, appointmentTime)

	text := fmt.Sprintf(
		"Kính gửi %s,\n\nĐây là email nhắc nhở về lịch hẹn của bạn.\n\nBác sĩ: %s\nNgày khám: %s\nGiờ khám: %s\n\nTrân trọng,\nDeepEyeX Medical Center",
		patientName, doctorName, appointmentDate, appointmentTime,
	)

	req := SendEmailRequest{
		From:    "DeepEyeX <onboard@resend.dev>",
		To:      []string{toEmail},
		Subject: "Nhắc nhở: Lịch hẹn khám bệnh sắp tới",
		HTML:    html,
		Text:    text,
	}

	_, err := s.SendEmail(req)
	return err
}

// SendPrescriptionEmail gửi email đơn thuốc
func (s *EmailService) SendPrescriptionEmail(toEmail, patientName, prescriptionDetails string) error {
	html := fmt.Sprintf(`
		<!DOCTYPE html>
		<html>
		<head>
			<meta charset="UTF-8">
			<style>
				body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
				.container { max-width: 600px; margin: 0 auto; padding: 20px; }
				.header { background-color: #2196F3; color: white; padding: 20px; text-align: center; }
				.content { padding: 20px; background-color: #f9f9f9; }
				.footer { text-align: center; padding: 20px; font-size: 12px; color: #666; }
				.prescription-box { background-color: white; padding: 15px; margin: 10px 0; border-left: 4px solid #2196F3; }
			</style>
		</head>
		<body>
			<div class="container">
				<div class="header">
					<h1>💊 Đơn Thuốc</h1>
				</div>
				<div class="content">
					<p>Kính gửi <strong>%s</strong>,</p>
					<p>Đơn thuốc của bạn đã được bác sĩ kê.</p>
					
					<div class="prescription-box">
						<h3>Chi tiết đơn thuốc:</h3>
						<pre>%s</pre>
					</div>
					
					<p><strong>Lưu ý:</strong> Vui lòng tuân thủ đúng liều lượng và thời gian dùng thuốc theo chỉ định của bác sĩ.</p>
					
					<p style="margin-top: 20px;">Trân trọng,<br><strong>DeepEyeX Medical Center</strong></p>
				</div>
				<div class="footer">
					<p>Email này được gửi tự động. Vui lòng không reply.</p>
					<p>&copy; 2025 DeepEyeX. All rights reserved.</p>
				</div>
			</div>
		</body>
		</html>
	`, patientName, prescriptionDetails)

	text := fmt.Sprintf(
		"Kính gửi %s,\n\nĐơn thuốc của bạn:\n\n%s\n\nLưu ý: Vui lòng tuân thủ đúng liều lượng và thời gian dùng thuốc theo chỉ định của bác sĩ.\n\nTrân trọng,\nDeepEyeX Medical Center",
		patientName, prescriptionDetails,
	)

	req := SendEmailRequest{
		From:    "DeepEyeX <onboard@resend.dev>",
		To:      []string{toEmail},
		Subject: "Đơn thuốc của bạn",
		HTML:    html,
		Text:    text,
	}

	_, err := s.SendEmail(req)
	return err
}


// formatCurrency format số tiền với dấu chấm ngăn cách hàng nghìn (VD: 2.500.000)
func formatCurrency(value float64) string {
	// Convert to int để bỏ phần thập phân
	intValue := int64(value)
	
	// Convert thành string
	str := fmt.Sprintf("%d", intValue)
	
	// Thêm dấu chấm ngăn cách từ phải sang trái
	var result string
	for i, digit := range str {
		if i > 0 && (len(str)-i)%3 == 0 {
			result += "."
		}
		result += string(digit)
	}
	
	return result
}
