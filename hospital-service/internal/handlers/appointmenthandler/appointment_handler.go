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

type CreateFollowUpRequest struct {
	PatientID       string   `json:"patient_id" binding:"required"`
	DoctorID        string   `json:"doctor_id" binding:"required"`
	HospitalID      string   `json:"hospital_id" binding:"required"`
	BookUserID      string   `json:"book_user_id" binding:"required"`
	SlotIDs         []string `json:"slot_ids" binding:"required"`
	Notes           string   `json:"notes"`
	ServiceName     string   `json:"service_name" binding:"required"`
	RelatedRecordID string   `json:"related_record_id" binding:"required"`
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
// @Summary Get appointments by patient ID with filters
// @Description Retrieve all appointments belonging to a specific patient with optional filters and sorting
// @Tags Appointments
// @Produce json
// @Param patient_id path string true "Patient ID"
// @Param status query string false "Filter by appointment status (PENDING/CONFIRMED/COMPLETED/CANCELED/PENDING_ONLINE/CONFIRMED_ONLINE/COMPLETED_ONLINE)"
// @Param date query string false "Filter by appointment date (format: YYYY-MM-DD)"
// @Param sort query string false "Sort by created date (newest/oldest, default: newest)"
// @Success 200 {array} appointment.Appointment
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /appointments/patient/{patient_id} [get]
func (h *AppointmentHandler) GetAppointmentsByPatient(c *gin.Context) {
	patientID := c.Param("patient_id")

	// Lấy query params
	status := c.Query("status")
	date := c.Query("date")
	sortBy := c.Query("sort")

	// Nếu có bất kỳ filter/sort params nào thì dùng method có filters
	if status != "" || date != "" || sortBy != "" {
		// Set default sort nếu không được cung cấp
		if sortBy == "" {
			sortBy = "newest"
		}
		appointments, err := h.service.GetByPatientIDWithFilters(patientID, status, date, sortBy)
		if err != nil {
			c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
			return
		}
		c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Appointments retrieved successfully", appointments))
		return
	}

	// Nếu không có filter thì dùng method cũ (backward compatible)
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

// @Summary Get today's appointments
// @Description Retrieve all appointments for today, optionally filtered by doctor_id, sorted by earliest timeslot and doctor ID
// @Tags Appointments
// @Produce json
// @Param doctor_id query string false "Doctor ID to filter appointments"
// @Success 200 {array} appointment.Appointment
// @Failure 500 {object} map[string]string
// @Router /appointments/today [get]
func (h *AppointmentHandler) GetTodayAppointments(c *gin.Context) {
	doctorID := c.Query("doctor_id") // Lấy doctor_id từ query param, có thể để trống
	if doctorID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "doctor_id is required"))
		return
	}
	appointments, err := h.service.GetTodayAppointments(doctorID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Today's appointments retrieved successfully", appointments))
}

// ---------------- Create Follow-Up Appointment ----------------
// @Summary Create a follow-up appointment
// @Description Create a new follow-up appointment linked to an existing medical record (relatedRecordID)
// @Tags Appointments
// @Accept json
// @Produce json
// @Param payload body CreateFollowUpRequest true "Follow-up appointment payload"
// @Success 201 {object} appointment.Appointment
// @Failure 400 {object} map[string]interface{} "Bad Request"
// @Failure 500 {object} map[string]interface{} "Internal Server Error"
// @Router /appointments/follow-up [post]
func (h *AppointmentHandler) CreateFollowUpAppointment(c *gin.Context) {
	var req CreateFollowUpRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	appt, err := h.service.CreateFollowUp(
		req.PatientID,
		req.DoctorID,
		req.HospitalID,
		req.BookUserID,
		req.SlotIDs,
		req.Notes,
		req.ServiceName,
		req.RelatedRecordID,
	)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "Follow-up appointment created successfully", appt))
}

// ---------------- Cancel Appointment ----------------
// @Summary Cancel an appointment
// @Description Cancel an appointment with time restriction (cannot cancel within 12 hours of appointment time)
// @Tags Appointments
// @Produce json
// @Param appointment_id path string true "Appointment ID"
// @Success 200 {object} map[string]string
// @Failure 400 {object} map[string]string
// @Failure 404 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /appointments/{appointment_id}/cancel [put]
func (h *AppointmentHandler) CancelAppointment(c *gin.Context) {
	appointmentID := c.Param("appointment_id")

	if err := h.service.CancelAppointment(appointmentID); err != nil {
		statusCode := http.StatusInternalServerError
		// Nếu là lỗi validation thời gian hoặc trạng thái thì trả về 400
		if err.Error() == "appointment is already canceled" ||
			err.Error() == "cannot cancel completed appointment" ||
			err.Error() == "appointment has no time slots" ||
			err.Error()[0:35] == "cannot cancel appointment within 12" {
			statusCode = http.StatusBadRequest
		} else if err.Error()[0:21] == "appointment not found" {
			statusCode = http.StatusNotFound
		}

		c.JSON(statusCode, utils.ErrorResponse(statusCode, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Appointment canceled successfully", nil))
}
