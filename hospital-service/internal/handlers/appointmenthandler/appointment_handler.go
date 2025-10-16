package appointmenthandler

import (
	"net/http"

	"hospital-service/internal/config"
	"hospital-service/internal/enums"
	"hospital-service/internal/models/appointment"
	"hospital-service/internal/services/appointmentservice"
	"hospital-service/internal/utils"

	"github.com/gin-gonic/gin"
)

type AppointmentHandler struct {
	service *appointmentservice.AppointmentService
	cfg     config.Config
}

type UpdateAppointmentStatusRequest struct {
	Status enums.AppointmentStatus `json:"status" binding:"required"`
}

type createAppointmentReq struct {
	PatientID  string `json:"patient_id" binding:"required"`
	DoctorID   string `json:"doctor_id" binding:"required"`
	HospitalID string `json:"hospital_id" binding:"required"`
	SlotID     string `json:"slot_id" binding:"required"`
	BookUserID string `json:"book_user_id" binding:"required"`
	Notes      string `json:"notes,omitempty"`
}

func NewAppointmentHandler(cfg config.Config, service *appointmentservice.AppointmentService) *AppointmentHandler {
	return &AppointmentHandler{service: service}
}

// ---------------- Get Appointment By ID ----------------
// @Summary Get appointment by ID
// @Description Retrieve an appointment record using its ID
// @Tags Appointments
// @Produce json
// @Param appointment_id path string true "Appointment ID"
// @Success 200 {object} appointment.Appointment
// @Failure 404 {object} map[string]string
// @Router /appointments/{appointment_id} [get]
func (h *AppointmentHandler) GetAppointmentByID(c *gin.Context) {
	id := c.Param("appointment_id")

	appt, err := h.service.GetByID(id)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Appointment retrieved successfully", appt))
}

// ---------------- Get Appointments By Patient ----------------
// @Summary Get appointments by patient ID
// @Description Retrieve all appointments belonging to a specific patient
// @Tags Appointments
// @Produce json
// @Param patient_id path string true "Patient ID"
// @Success 200 {array} appointment.Appointment
// @Failure 500 {object} map[string]string
// @Router /appointments/patient/{patient_id} [get]
func (h *AppointmentHandler) GetAppointmentsByPatient(c *gin.Context) {
	patientID := c.Param("patient_id")

	appointments, err := h.service.GetByPatientID(patientID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Appointments retrieved successfully", appointments))
}

// ---------------- Get Appointments By Doctor ----------------
// @Summary Get appointments by doctor ID
// @Description Retrieve all appointments assigned to a specific doctor
// @Tags Appointments
// @Produce json
// @Param doctor_id path string true "Doctor ID"
// @Success 200 {array} appointment.Appointment
// @Failure 500 {object} map[string]string
// @Router /appointments/doctor/{doctor_id} [get]
func (h *AppointmentHandler) GetAppointmentsByDoctor(c *gin.Context) {
	doctorID := c.Param("doctor_id")

	appointments, err := h.service.GetByDoctorID(doctorID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Appointments retrieved successfully", appointments))
}

// ---------------- Update Appointment Status ----------------
// @Summary Update appointment status
// @Description Update the status of an appointment (PENDING, CONFIRMED, COMPLETED, CANCELED)
// @Tags Appointments
// @Accept json
// @Produce json
// @Param appointment_id path string true "Appointment ID"
// @Param status body UpdateAppointmentStatusRequest true "New status"
// @Success 200 {object} map[string]string
// @Failure 400 {object} map[string]string
// @Router /appointments/{appointment_id}/status [put]
func (h *AppointmentHandler) UpdateAppointmentStatus(c *gin.Context) {
	id := c.Param("appointment_id")

	var req UpdateAppointmentStatusRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	if err := h.service.UpdateStatus(id, req.Status); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "appointment status updated successfully", nil))
}

// ---------------- Update Appointment Detail ----------------
// @Summary Update appointment details
// @Description Update the details of an existing appointment (date, notes, etc.)
// @Tags Appointments
// @Accept json
// @Produce json
// @Param appointment_id path string true "Appointment ID"
// @Param appointment body appointment.Appointment true "Updated appointment data"
// @Success 200 {object} map[string]string
// @Failure 400 {object} map[string]string
// @Router /appointments/{appointment_id}/detail [put]
func (h *AppointmentHandler) UpdateAppointmentDetail(c *gin.Context) {
	id := c.Param("appointment_id")

	var updated appointment.Appointment
	if err := c.ShouldBindJSON(&updated); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	if err := h.service.UpdateDetail(id, &updated); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "appointment details updated successfully", nil))
}

// ---------------- List All Appointments ----------------
// @Summary List all appointments
// @Description Retrieve a list of all appointments in the system with optional filters
// @Tags Appointments
// @Produce json
// @Param patient_name query string false "Filter by patient name (partial match)"
// @Param status query string false "Filter by appointment status (PENDING/CONFIRMED/COMPLETED/CANCELLED)"
// @Param doctor_id query string false "Filter by doctor ID (exact match)"
// @Success 200 {array} appointment.Appointment
// @Failure 500 {object} map[string]string
// @Router /appointments [get]
func (h *AppointmentHandler) ListAllAppointments(c *gin.Context) {
	// Lấy query params
	patientName := c.Query("patient_name")
	status := c.Query("status")
	doctorID := c.Query("doctor_id")

	appointments, err := h.service.ListAll(patientName, status, doctorID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Appointments retrieved successfully", appointments))
}

// ---------------- Delete Appointment ----------------
// @Summary Delete an appointment
// @Description Remove an appointment by its ID
// @Tags Appointments
// @Produce json
// @Param appointment_id path string true "Appointment ID"
// @Success 200 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /appointments/{appointment_id} [delete]
func (h *AppointmentHandler) DeleteAppointment(c *gin.Context) {
	id := c.Param("appointment_id")

	if err := h.service.Delete(id); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Appointment deleted successfully", nil))
}

// ---------------- Get Online Appointments ----------------
// @Summary Get online appointments
// @Description Get all online appointments (status = PendingOnline) by bookUserID or doctorID
// @Tags Appointments
// @Produce json
// @Param book_user_id query string false "Book user ID"
// @Param doctor_id query string false "Doctor ID"
// @Success 200 {array} appointment.Appointment
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /appointments/online [get]
func (h *AppointmentHandler) GetOnlineAppointments(c *gin.Context) {
	bookUserID := c.Query("book_user_id")
	doctorID := c.Query("doctor_id")

	if bookUserID == "" && doctorID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "book_user_id or doctor_id is required"))
		return
	}

	appointments, err := h.service.GetOnlineAppointments(bookUserID, doctorID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Online appointments retrieved successfully", appointments))
}
