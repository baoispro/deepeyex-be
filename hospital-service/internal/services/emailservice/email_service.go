package emailservice

import (
	"fmt"
	"hospital-service/internal/config"
	"hospital-service/internal/services/notificationservice"
	"hospital-service/internal/websocket"

	"github.com/resend/resend-go/v2"
)

type EmailService struct {
	client *resend.Client
	cfg    config.Config
	wsHub  *websocket.Hub
	notificationSvc    *notificationservice.NotificationService

}

// NewEmailService khởi tạo email service mới
func NewEmailService(cfg config.Config, wsHub *websocket.Hub, notificationSvc *notificationservice.NotificationService) *EmailService {
	client := resend.NewClient(cfg.ResendAPIKey)
	return &EmailService{
		client: client,
		cfg:    cfg,
		wsHub:  wsHub,
		notificationSvc: notificationSvc,
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
	ServiceID string  `json:"service_id"`
	ItemName  string  `json:"item_name"`
	Quantity  int     `json:"quantity"`
	Price     float64 `json:"price"`
}

// OrderConfirmationItem struct cho email xác nhận đơn hàng
type OrderConfirmationItem struct {
	DrugID    string   `json:"drug_id"`
	Name      string   `json:"name"`
	Image     string   `json:"image,omitempty"`
	Price     float64  `json:"price"`
	SalePrice *float64 `json:"sale_price,omitempty"`
	Quantity  int      `json:"quantity"`
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
				.header { background-color: #1250dc; color: white; padding: 20px; text-align: center; }
				.content { padding: 20px; background-color: #f9f9f9; }
				.footer { text-align: center; padding: 20px; font-size: 12px; color: #666; }
				.info-box { background-color: white; padding: 15px; margin: 10px 0; border-left: 4px solid #1250dc; }
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


// SendOrderConfirmation gửi email xác nhận đơn hàng thành công
func (s *EmailService) SendOrderConfirmation(
	toEmail, patientName, orderCode string,
	orderItems []OrderConfirmationItem,
	deliveryMethod, deliveryAddress, deliveryPhone, deliveryFullname string,
	deliveryCity, deliveryDistrict, deliveryWard, deliveryNotes string,
	deliveryFee float64,
) error {

	// Tính tổng tiền sản phẩm
	var subtotal float64
	for _, item := range orderItems {
		// Sử dụng giá sale nếu có, không thì dùng giá gốc
		finalPrice := item.Price
		if item.SalePrice != nil && *item.SalePrice > 0 {
			finalPrice = *item.SalePrice
		}
		subtotal += finalPrice * float64(item.Quantity)
	}

	// Tổng tiền cuối cùng (bao gồm phí ship)
	total := subtotal + deliveryFee

	// Tạo danh sách sản phẩm HTML
	var orderDetailsHTML string
	for _, item := range orderItems {
		// Sử dụng giá sale nếu có
		finalPrice := item.Price
		priceDisplay := formatCurrency(item.Price)
		
		if item.SalePrice != nil && *item.SalePrice > 0 {
			finalPrice = *item.SalePrice
			// Hiển thị giá gốc gạch ngang + giá sale
			priceDisplay = fmt.Sprintf(`<span style="text-decoration: line-through; color: #999;">%s₫</span> <span style="color: #e53935; font-weight: bold;">%s₫</span>`, 
				formatCurrency(item.Price), 
				formatCurrency(*item.SalePrice))
		} else {
			priceDisplay = fmt.Sprintf(`%s₫`, formatCurrency(item.Price))
		}
		
		orderDetailsHTML += fmt.Sprintf(
			`<tr>
				<td style="padding: 10px; border-bottom: 1px solid #eee;">
					<div style="display: flex; align-items: center; gap: 10px;">
						%s
						<span>%s</span>
					</div>
				</td>
				<td style="padding: 10px; border-bottom: 1px solid #eee; text-align: center;">%d</td>
				<td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right;">%s</td>
				<td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right; font-weight: bold;">%s₫</td>
			</tr>`,
			func() string {
				if item.Image != "" {
					return fmt.Sprintf(`<img src="%s" alt="%s" style="width: 50px; height: 50px; object-fit: cover; border-radius: 4px;">`, item.Image, item.Name)
				}
				return ""
			}(),
			item.Name,
			item.Quantity,
			priceDisplay,
			formatCurrency(finalPrice*float64(item.Quantity)),
		)
	}

	// Tạo thông tin giao hàng HTML
	var deliveryMethodText string
	var deliveryInfoHTML string

	if deliveryMethod == "HOME_DELIVERY" {
		deliveryMethodText = "Giao hàng tận nơi"
		fullAddress := deliveryAddress
		if deliveryWard != "" {
			fullAddress = deliveryWard + ", " + fullAddress
		}
		if deliveryDistrict != "" {
			fullAddress = deliveryDistrict + ", " + fullAddress
		}
		if deliveryCity != "" {
			fullAddress = deliveryCity + ", " + fullAddress
		}

		deliveryInfoHTML = fmt.Sprintf(`
			<div class="info-box">
				<h3>Thông tin giao hàng:</h3>
				<p><strong>Người nhận:</strong> %s</p>
				<p><strong>Số điện thoại:</strong> %s</p>
				<p><strong>Địa chỉ:</strong> %s</p>
				<p><strong>Phí vận chuyển:</strong> %s₫</p>
				%s
			</div>`,
			deliveryFullname,
			deliveryPhone,
			fullAddress,
			formatCurrency(deliveryFee),
			func() string {
				if deliveryNotes != "" {
					return fmt.Sprintf("<p><strong>Ghi chú:</strong> %s</p>", deliveryNotes)
				}
				return ""
			}(),
		)
	} else {
		deliveryMethodText = "Nhận tại bệnh viện"
		deliveryInfoHTML = `
			<div class="info-box">
				<h3>Thông tin nhận hàng:</h3>
				<p><strong>Phương thức:</strong> Nhận tại bệnh viện</p>
				<p>Vui lòng đến quầy lễ tân để nhận đơn hàng của bạn.</p>
			</div>`
	}

	html := fmt.Sprintf(`
		<!DOCTYPE html>
		<html>
		<head>
			<meta charset="UTF-8">
			<style>
				body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
				.container { max-width: 600px; margin: 0 auto; padding: 20px; }
				.header { background-color: #1250dc; color: white; padding: 20px; text-align: center; }
				.content { padding: 20px; background-color: #f9f9f9; }
				.footer { text-align: center; padding: 20px; font-size: 12px; color: #666; }
				.info-box { background-color: white; padding: 15px; margin: 10px 0; border-left: 4px solid #1250dc; }
				table { width: 100%%; border-collapse: collapse; background-color: white; margin: 10px 0; }
				th { background-color: #1250dc; color: white; padding: 10px; text-align: left; }
				.total-row { background-color: #f0f0f0; font-weight: bold; }
				.success-badge { background-color: #1250dc; color: white; padding: 5px 15px; border-radius: 20px; display: inline-block; }
			</style>
		</head>
		<body>
			<div class="container">
				<div class="header">
					<h1>✅ Đặt Hàng Thành Công</h1>
				</div>
				<div class="content">
					<p>Kính gửi <strong>%s</strong>,</p>
					<p>Cảm ơn bạn đã đặt hàng tại <strong>DeepEyeX Medical Center</strong>!</p>
					<p>Đơn hàng của bạn đã được xác nhận và đang được xử lý.</p>
					
					<div class="info-box">
						<h3>Thông tin đơn hàng:</h3>
						<p><strong>Mã đơn hàng:</strong> %s</p>
						<p><strong>Phương thức nhận hàng:</strong> <span class="success-badge">%s</span></p>
					</div>

					<h3>Chi tiết sản phẩm:</h3>
					<table>
						<thead>
							<tr>
								<th>Sản phẩm</th>
								<th style="text-align: center;">Số lượng</th>
								<th style="text-align: right;">Đơn giá</th>
								<th style="text-align: right;">Thành tiền</th>
							</tr>
						</thead>
						<tbody>
							%s
							<tr class="total-row">
								<td colspan="3" style="padding: 10px; text-align: right;">Tạm tính:</td>
								<td style="padding: 10px; text-align: right;">%s₫</td>
							</tr>
							<tr class="total-row">
								<td colspan="3" style="padding: 10px; text-align: right;">Phí vận chuyển:</td>
								<td style="padding: 10px; text-align: right;">%s₫</td>
							</tr>
							<tr class="total-row" style="background-color: #1250dc; color: white;">
								<td colspan="3" style="padding: 15px; text-align: right; font-size: 18px;">TỔNG CỘNG:</td>
								<td style="padding: 15px; text-align: right; font-size: 18px;">%s₫</td>
							</tr>
						</tbody>
					</table>

					%s
					
					<p style="margin-top: 20px;">Nếu có bất kỳ thắc mắc nào, vui lòng liên hệ với chúng tôi.</p>
					<p style="margin-top: 20px;">Trân trọng,<br><strong>DeepEyeX Medical Center</strong></p>
				</div>
				<div class="footer">
					<p>Email này được gửi tự động. Vui lòng không reply.</p>
					<p>&copy; 2025 DeepEyeX. All rights reserved.</p>
				</div>
			</div>
		</body>
		</html>
	`,
		patientName,
		orderCode,
		deliveryMethodText,
		orderDetailsHTML,
		formatCurrency(subtotal),
		formatCurrency(deliveryFee),
		formatCurrency(total),
		deliveryInfoHTML,
	)

	text := fmt.Sprintf(
		"Kính gửi %s,\n\nĐơn hàng của bạn đã được đặt thành công!\n\nMã đơn hàng: %s\nPhương thức: %s\nTổng tiền: %s₫\n\nTrân trọng,\nDeepEyeX Medical Center",
		patientName, orderCode, deliveryMethodText, formatCurrency(total),
	)

	req := SendEmailRequest{
		From:    "DeepEyeX <onboard@resend.dev>",
		To:      []string{toEmail},
		Subject: "Xác nhận đơn hàng - " + orderCode,
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

// SendAppointmentCancelNotification gửi email thông báo hủy lịch khám
func (s *EmailService) SendAppointmentCancelNotification(toEmail, patientName, doctorName, appointmentDate, appointmentTime, reason, patientID string) error {
	html := fmt.Sprintf(`
		<!DOCTYPE html>
		<html>
		<head>
			<meta charset="UTF-8">
			<style>
				body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
				.container { max-width: 600px; margin: 0 auto; padding: 20px; }
				.header { background-color: #dc3545; color: white; padding: 20px; text-align: center; }
				.content { padding: 20px; background-color: #f9f9f9; }
				.footer { text-align: center; padding: 20px; font-size: 12px; color: #666; }
				.info-box { background-color: white; padding: 15px; margin: 10px 0; border-left: 4px solid #dc3545; }
				.warning { background-color: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 5px; margin: 15px 0; }
			</style>
		</head>
		<body>
			<div class="container">
				<div class="header">
					<h1>⚠️ Thông Báo Hủy Lịch Hẹn</h1>
				</div>
				<div class="content">
					<p>Kính gửi <strong>%s</strong>,</p>
					<p>Chúng tôi rất tiếc phải thông báo rằng lịch hẹn khám bệnh của bạn đã bị hủy.</p>
					
					<div class="info-box">
						<h3>Thông tin lịch hẹn đã hủy:</h3>
						<p><strong>Bác sĩ:</strong> %s</p>
						<p><strong>Ngày khám:</strong> %s</p>
						<p><strong>Giờ khám:</strong> %s</p>
					</div>
					
					<div class="warning">
						<p><strong>⚠️ Lý do hủy:</strong></p>
						<p>%s</p>
					</div>
					
					<p>Nếu bạn cần hỗ trợ hoặc muốn đặt lịch hẹn mới, vui lòng liên hệ với chúng tôi.</p>
					<p>Chúng tôi xin lỗi vì sự bất tiện này.</p>
					
					<p style="margin-top: 20px;">Trân trọng,<br><strong>DeepEyeX Medical Center</strong></p>
				</div>
				<div class="footer">
					<p>Email này được gửi tự động. Vui lòng không reply.</p>
					<p>&copy; 2025 DeepEyeX. All rights reserved.</p>
				</div>
			</div>
		</body>
		</html>
	`, patientName, doctorName, appointmentDate, appointmentTime, reason)

	text := fmt.Sprintf(
		"Kính gửi %s,\n\nChúng tôi rất tiếc phải thông báo rằng lịch hẹn khám bệnh của bạn đã bị hủy.\n\n"+
			"Bác sĩ: %s\nNgày khám: %s\nGiờ khám: %s\nLý do: %s\n\n"+
			"Nếu bạn cần hỗ trợ hoặc muốn đặt lịch hẹn mới, vui lòng liên hệ với chúng tôi.\n\nTrân trọng,\nDeepEyeX Medical Center",
		patientName, doctorName, appointmentDate, appointmentTime, reason,
	)

	req := SendEmailRequest{
		From:    "DeepEyeX <onboard@resend.dev>",
		To:      []string{toEmail},
		Subject: "Thông báo hủy lịch hẹn",
		HTML:    html,
		Text:    text,
	}

	_, err := s.SendEmail(req)
	
	// Gửi WebSocket notification cho bệnh nhân (async)
	if s.wsHub != nil && patientID != "" {
		go s.sendCancelWebSocketNotification(patientID, patientName, doctorName, appointmentDate, appointmentTime, reason)
	}

	if s.notificationSvc != nil {
		go func() {
			_, _ = s.notificationSvc.CreateNotification(
				patientID,
				"Lịch hẹn đã bị hủy",
				"Lịch hẹn của bạn đã bị hủy",
				fmt.Sprintf("/patient/appointments/%s", patientID),
			)
		}()
	}
	
	return err
}

// sendCancelWebSocketNotification gửi WebSocket notification khi hủy lịch
func (s *EmailService) sendCancelWebSocketNotification(patientID, patientName, doctorName, appointmentDate, appointmentTime, reason string) {
	payload := map[string]interface{}{
		"message":           "Lịch hẹn của bạn đã bị hủy",
		"doctor_name":       doctorName,
		"appointment_date":  appointmentDate,
		"appointment_time":  appointmentTime,
		"reason":            reason,
		"notification_type": "APPOINTMENT_CANCELLED",
	}

	// Broadcast WebSocket notification đến patient
	s.wsHub.BroadcastToPatient(patientID, websocket.CancelAppointment, payload)
	fmt.Printf("[Email Service] WebSocket notification sent to patient %s\n", patientID)
}

// SendFollowUpConfirmationEmail gửi email xác nhận lịch tái khám
func (s *EmailService) SendFollowUpConfirmationEmail(toEmail, patientName, doctorName, doctorFullName, hospitalName, confirmationLink, appointmentDate, appointmentTime string) error {
	html := fmt.Sprintf(`
		<!DOCTYPE html>
		<html>
		<head>
			<meta charset="UTF-8">
			<style>
			body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
			.container { max-width: 600px; margin: 0 auto; padding: 20px; }
			.header { background-color: #1250dc; color: white; padding: 20px; text-align: center; }
			.content { padding: 20px; background-color: #f9f9f9; }
			.footer { text-align: center; padding: 20px; font-size: 12px; color: #666; }
			.info-box { background-color: white; padding: 15px; margin: 10px 0; border-left: 4px solid #1250dc; }
			.button { display: inline-block; background-color: #1250dc !important; color: #ffffff !important; padding: 12px 30px; text-decoration: none !important; border-radius: 5px; margin: 20px 0; font-weight: bold; border: none; text-decoration: none; }
			.expires { background-color: #fff3cd; border: 1px solid #ffc107; padding: 10px; border-radius: 5px; margin: 15px 0; }
			</style>
		</head>
		<body>
			<div class="container">
				<div class="header">
					<h1>📅 Xác Nhận Lịch Tái Khám</h1>
				</div>
				<div class="content">
					<p>Kính gửi <strong>%s</strong>,</p>
					<p>Bạn có một lịch tái khám đang chờ xác nhận.</p>
					
					<div class="info-box">
						<h3>Thông tin lịch tái khám:</h3>
						<p><strong>Bác sĩ:</strong> %s</p>
						<p><strong>Bệnh viện:</strong> %s</p>
						<p><strong>Ngày khám:</strong> %s</p>
						<p><strong>Giờ khám:</strong> %s</p>
					</div>
					
					<div style="text-align: center; margin: 30px 0;">
						<a href="%s" style="display: inline-block; background-color: #1250dc !important; color: #ffffff !important; padding: 12px 30px; text-decoration: none !important; border-radius: 5px; margin: 20px 0; font-weight: bold;">Xác Nhận Lịch Hẹn</a>
					</div>
					
					<div class="expires">
						<p><strong>⚠️ Lưu ý:</strong> Vui lòng xác nhận trong vòng 7 ngày. Link xác nhận sẽ hết hạn sau thời gian này.</p>
					</div>
					
					<p>Nếu bạn không yêu cầu lịch tái khám này, vui lòng bỏ qua email này.</p>
					
					<p style="margin-top: 20px;">Trân trọng,<br><strong>DeepEyeX Medical Center</strong></p>
				</div>
				<div class="footer">
					<p>Email này được gửi tự động. Vui lòng không reply.</p>
					<p>&copy; 2025 DeepEyeX. All rights reserved.</p>
				</div>
			</div>
		</body>
		</html>
	`, patientName, doctorFullName, hospitalName, appointmentDate, appointmentTime, confirmationLink)

	text := fmt.Sprintf(
		"Kính gửi %s,\n\nBạn có một lịch tái khám đang chờ xác nhận.\n\n"+
			"Bác sĩ: %s\nBệnh viện: %s\nNgày khám: %s\nGiờ khám: %s\n\n"+
			"Vui lòng click vào link sau để xác nhận:\n%s\n\n"+
			"Link xác nhận sẽ hết hạn trong 7 ngày.\n\n"+
			"Trân trọng,\nDeepEyeX Medical Center",
		patientName, doctorFullName, hospitalName, appointmentDate, appointmentTime, confirmationLink,
	)

	req := SendEmailRequest{
		From:    "DeepEyeX <onboard@resend.dev>",
		To:      []string{toEmail},
		Subject: "Xác nhận lịch tái khám",
		HTML:    html,
		Text:    text,
	}

	_, err := s.SendEmail(req)
	return err
}
