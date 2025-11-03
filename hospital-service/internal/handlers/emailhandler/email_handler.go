package emailhandler

import (
	"fmt"
	"hospital-service/internal/services/appointmentservice"
	"hospital-service/internal/services/emailservice"
	"hospital-service/internal/utils"
	"net/http"

	"github.com/gin-gonic/gin"
)

type EmailHandler struct {
	service            *emailservice.EmailService
	appointmentService appointmentservice.AppointmentServiceInterface
}

// NewEmailHandler khởi tạo handler mới
func NewEmailHandler(service *emailservice.EmailService) *EmailHandler {
	return &EmailHandler{service: service}
}

// SetAppointmentService set appointment service
func (h *EmailHandler) SetAppointmentService(appointmentSvc appointmentservice.AppointmentServiceInterface) {
	h.appointmentService = appointmentSvc
}

// SendEmailRequest cấu trúc request cho API gửi email
type sendEmailAPIRequest struct {
	From    string   `json:"from" binding:"required"`
	To      []string `json:"to" binding:"required"`
	Subject string   `json:"subject" binding:"required"`
	HTML    string   `json:"html" binding:"required"`
	Text    string   `json:"text,omitempty"`
}



// SendAppointmentReminderRequest cấu trúc request cho email nhắc nhở lịch hẹn
type sendAppointmentReminderRequest struct {
	ToEmail         string `json:"to_email" binding:"required,email"`
	PatientName     string `json:"patient_name" binding:"required"`
	DoctorName      string `json:"doctor_name" binding:"required"`
	AppointmentDate string `json:"appointment_date" binding:"required"`
	AppointmentTime string `json:"appointment_time" binding:"required"`
}

// SendPrescriptionRequest cấu trúc request cho email đơn thuốc
type sendPrescriptionRequest struct {
	ToEmail             string `json:"to_email" binding:"required,email"`
	PatientName         string `json:"patient_name" binding:"required"`
	PrescriptionDetails string `json:"prescription_details" binding:"required"`
}

// SendOrderConfirmationRequest cấu trúc request cho email xác nhận đơn hàng
type sendOrderConfirmationRequest struct {
	ToEmail          string                               `json:"to_email" binding:"required,email"`
	PatientName      string                               `json:"patient_name" binding:"required"`
	OrderCode        string                               `json:"order_code" binding:"required"`
	OrderItems       []emailservice.OrderConfirmationItem `json:"order_items" binding:"required,dive"`
	DeliveryMethod   string                               `json:"delivery_method" binding:"required"`
	DeliveryAddress  string                               `json:"delivery_address,omitempty"`
	DeliveryPhone    string                               `json:"delivery_phone,omitempty"`
	DeliveryFullname string                               `json:"delivery_fullname,omitempty"`
	DeliveryCity     string                               `json:"delivery_city,omitempty"`
	DeliveryDistrict string                               `json:"delivery_district,omitempty"`
	DeliveryWard     string                               `json:"delivery_ward,omitempty"`
	DeliveryNotes    string                               `json:"delivery_notes,omitempty"`
	DeliveryFee      float64                              `json:"delivery_fee"`
}

// SendEmail gửi email tùy chỉnh
// @Summary Send custom email
// @Description Send a custom email to recipients
// @Tags Email
// @Accept json
// @Produce json
// @Param request body sendEmailAPIRequest true "Email details"
// @Success 200 {object} utils.APIResponse
// @Failure 400 {object} utils.APIResponse
// @Failure 500 {object} utils.APIResponse
// @Router /emails/send [post]
func (h *EmailHandler) SendEmail(c *gin.Context) {
	var req sendEmailAPIRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	emailReq := emailservice.SendEmailRequest{
		From:    req.From,
		To:      req.To,
		Subject: req.Subject,
		HTML:    req.HTML,
		Text:    req.Text,
	}

	emailID, err := h.service.SendEmail(emailReq)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Email sent successfully", gin.H{
		"email_id": emailID,
	}))
}


// sendAppointmentConfirmationRequest đại diện cho body gửi từ client
type sendAppointmentConfirmationRequest struct {
	ToEmail         string       `json:"to_email" binding:"required"`
	PatientName     string       `json:"patient_name" binding:"required"`
	DoctorName      string       `json:"doctor_name" binding:"required"`
	AppointmentDate string       `json:"appointment_date" binding:"required"`
	AppointmentTime string       `json:"appointment_time" binding:"required"`
	AppointmentCode string       `json:"appointment_code" binding:"required"`
	OrderItems      []emailservice.OrderItem  `json:"order_items" binding:"required,dive"`
}


// SendAppointmentConfirmation gửi email xác nhận lịch hẹn
// @Summary Send appointment confirmation email
// @Description Send confirmation email for a booked appointment
// @Tags Email
// @Accept json
// @Produce json
// @Param request body sendAppointmentConfirmationRequest true "Appointment details"
// @Success 200 {object} utils.APIResponse
// @Failure 400 {object} utils.APIResponse
// @Failure 500 {object} utils.APIResponse
// @Router /emails/appointment-confirmation [post]
func (h *EmailHandler) SendAppointmentConfirmation(c *gin.Context) {
	var req sendAppointmentConfirmationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	err := h.service.SendAppointmentConfirmation(
		req.ToEmail,
		req.PatientName,
		req.DoctorName,
		req.AppointmentDate,
		req.AppointmentTime,
		req.AppointmentCode,
		req.OrderItems, 
	)

	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Appointment confirmation email sent successfully", nil))
}


// SendAppointmentReminder gửi email nhắc nhở lịch hẹn
// @Summary Send appointment reminder email
// @Description Send reminder email for an upcoming appointment
// @Tags Email
// @Accept json
// @Produce json
// @Param request body sendAppointmentReminderRequest true "Appointment details"
// @Success 200 {object} utils.APIResponse
// @Failure 400 {object} utils.APIResponse
// @Failure 500 {object} utils.APIResponse
// @Router /emails/appointment-reminder [post]
func (h *EmailHandler) SendAppointmentReminder(c *gin.Context) {
	var req sendAppointmentReminderRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	err := h.service.SendAppointmentReminder(
		req.ToEmail,
		req.PatientName,
		req.DoctorName,
		req.AppointmentDate,
		req.AppointmentTime,
	)

	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Appointment reminder email sent successfully", nil))
}

// SendPrescription gửi email đơn thuốc
// @Summary Send prescription email
// @Description Send prescription details via email
// @Tags Email
// @Accept json
// @Produce json
// @Param request body sendPrescriptionRequest true "Prescription details"
// @Success 200 {object} utils.APIResponse
// @Failure 400 {object} utils.APIResponse
// @Failure 500 {object} utils.APIResponse
// @Router /emails/prescription [post]
func (h *EmailHandler) SendPrescription(c *gin.Context) {
	var req sendPrescriptionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	err := h.service.SendPrescriptionEmail(
		req.ToEmail,
		req.PatientName,
		req.PrescriptionDetails,
	)

	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Prescription email sent successfully", nil))
}

// SendOrderConfirmation gửi email xác nhận đơn hàng
// @Summary Send order confirmation email
// @Description Send confirmation email for a successful order with full delivery information
// @Tags Email
// @Accept json
// @Produce json
// @Param request body sendOrderConfirmationRequest true "Order details"
// @Success 200 {object} utils.APIResponse
// @Failure 400 {object} utils.APIResponse
// @Failure 500 {object} utils.APIResponse
// @Router /emails/order-confirmation [post]
func (h *EmailHandler) SendOrderConfirmation(c *gin.Context) {
	var req sendOrderConfirmationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	err := h.service.SendOrderConfirmation(
		req.ToEmail,
		req.PatientName,
		req.OrderCode,
		req.OrderItems,
		req.DeliveryMethod,
		req.DeliveryAddress,
		req.DeliveryPhone,
		req.DeliveryFullname,
		req.DeliveryCity,
		req.DeliveryDistrict,
		req.DeliveryWard,
		req.DeliveryNotes,
		req.DeliveryFee,
	)

	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Order confirmation email sent successfully", nil))
}

// SendCancelNotificationRequest request để gửi email hủy lịch
type SendCancelNotificationRequest struct {
	AppointmentID string `json:"appointment_id" binding:"required"`
	PatientEmail  string `json:"patient_email" binding:"required,email"`
	PatientID     string `json:"patient_id" binding:"required"`
	PatientName   string `json:"patient_name" binding:"required"`
	DoctorID      string `json:"doctor_id" binding:"required"`     // ✅ Thêm để tạo pending
	DoctorName    string `json:"doctor_name" binding:"required"`
	HospitalID    string `json:"hospital_id" binding:"required"`   // ✅ Thêm để tạo pending
	ServiceName   string `json:"service_name" binding:"required"`  // ✅ Thêm để tạo pending
	AppointmentDate string `json:"appointment_date" binding:"required"`
	AppointmentTime string `json:"appointment_time" binding:"required"`
	Reason        string `json:"reason" binding:"required"`
}

// SendCancelNotification gửi email thông báo hủy lịch + notification
// @Summary Send cancel notification email
// @Description Send email and notification to patient when appointment is canceled
// @Tags Email
// @Accept json
// @Produce json
// @Param request body SendCancelNotificationRequest true "Cancel notification details"
// @Success 200 {object} utils.APIResponse
// @Failure 400 {object} utils.APIResponse
// @Failure 500 {object} utils.APIResponse
// @Router /emails/send-cancel-notification [post]
func (h *EmailHandler) SendCancelNotification(c *gin.Context) {
	var req SendCancelNotificationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	// ✅ Tạo pending appointment với slot gần nhất TRƯỚC
	var pendingInfo *emailservice.PendingAppointmentInfo
	if h.appointmentService != nil {
		pendingAppt, err := h.appointmentService.CreatePendingAppointmentAfterCancel(
			req.PatientID,
			req.DoctorID,
			req.HospitalID,
			req.ServiceName,
		)
		if err != nil {
			c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Failed to create pending appointment", gin.H{
				"patient_id": req.PatientID,
				"warning":    err.Error(),
			}))
			return
		}
		
		// Convert PendingFollowUpAppointment sang PendingAppointmentInfo
		if pendingAppt != nil {
			pendingInfo = &emailservice.PendingAppointmentInfo{
				ConfirmationToken: pendingAppt.ConfirmationToken,
				AppointmentDate:   "",
				AppointmentTime:   "",
				ExpiresAt:         pendingAppt.ExpiresAt.Format("02/01/2006 15:04"),
			}
			
			// Set date and time from suggested slot times
			if pendingAppt.SuggestedStartTime != nil {
				pendingInfo.AppointmentDate = pendingAppt.SuggestedStartTime.Format("02/01/2006")
			}
			if pendingAppt.SuggestedStartTime != nil && pendingAppt.SuggestedEndTime != nil {
				pendingInfo.AppointmentTime = fmt.Sprintf("%s - %s", 
					pendingAppt.SuggestedStartTime.Format("15:04"), 
					pendingAppt.SuggestedEndTime.Format("15:04"))
			}
		}
	}

	// Gửi email thông báo hủy với thông tin pending (nếu có)
	// Email sẽ có cả nút confirm nếu tạo pending thành công
	err := h.service.SendAppointmentCancelNotification(
		req.PatientEmail,
		req.PatientName,
		req.DoctorName,
		req.AppointmentDate,
		req.AppointmentTime,
		req.Reason,
		req.PatientID,
		pendingInfo, // ✅ Truyền pending info vào
	)

	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Cancel notification and pending appointment created successfully", gin.H{
		"patient_id": req.PatientID,
		"message":    "Email sent and pending appointment created",
	}))
}
