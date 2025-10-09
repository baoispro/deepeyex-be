package emailhandler

import (
	"hospital-service/internal/services/emailservice"
	"hospital-service/internal/utils"
	"net/http"

	"github.com/gin-gonic/gin"
)

type EmailHandler struct {
	service *emailservice.EmailService
}

// NewEmailHandler khởi tạo handler mới
func NewEmailHandler(service *emailservice.EmailService) *EmailHandler {
	return &EmailHandler{service: service}
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

