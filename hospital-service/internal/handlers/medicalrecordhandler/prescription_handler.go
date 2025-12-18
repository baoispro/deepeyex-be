package medicalrecordhandler

import (
	"hospital-service/internal/config"
	"hospital-service/internal/services/medicalrecordservice"
	"hospital-service/internal/utils"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

// PrescriptionHandler quản lý API cho đơn thuốc
type PrescriptionHandler struct {
	service *medicalrecordservice.PrescriptionService
	cfg     config.Config
}

// NewPrescriptionHandler khởi tạo PrescriptionHandler
func NewPrescriptionHandler(cfg config.Config, service *medicalrecordservice.PrescriptionService) *PrescriptionHandler {
	return &PrescriptionHandler{service: service, cfg: cfg}
}

// -------------------- Request Structs --------------------

// Request để tạo mới Prescription
type createPrescriptionReq struct {
	Status     string     `json:"status" binding:"required"`
	RecordID   string     `json:"record_id" binding:"required"`
	ApprovedBy string     `json:"approved_by"`
	ApprovedAt *time.Time `json:"approved_at"`
}

// Request để update Prescription
type updatePrescriptionReq struct {
	Status     string     `json:"status"`
	ApprovedBy string     `json:"approved_by"`
	ApprovedAt *time.Time `json:"approved_at"`
}

// CreatePrescription godoc
// @Summary Thêm toa thuốc
// @Description Tạo một toa thuốc mới cho patient / medical record
// @Tags Prescriptions
// @Accept json
// @Produce json
// @Param data body medicalrecordservice.PrescriptionRequest true "Prescription Data"
// @Success 201 {object} medicalrecord.Prescription
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /prescriptions [post]
// func (h *PrescriptionHandler) CreatePrescription(c *gin.Context) {
// 	var req medicalrecordservice.PrescriptionRequest
// 	if err := c.ShouldBindJSON(&req); err != nil {
// 		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
// 		return
// 	}

// 	prescription, err := h.service.CreatePrescription(&req)
// 	if err != nil {
// 		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
// 		return
// 	}

// 	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "Prescription created successfully", prescription))
// }

// -------------------- Get Prescription By ID --------------------
// @Summary Get prescription by ID
// @Description Retrieve a prescription by its ID
// @Tags Prescriptions
// @Produce json
// @Param prescription_id path string true "Prescription ID"
// @Success 200 {object}  medicalrecord.Prescription
// @Failure 404 {object} map[string]string
// @Router /prescriptions/{prescription_id} [get]
func (h *PrescriptionHandler) GetPrescriptionByID(c *gin.Context) {
	id := c.Param("prescription_id")
	prescription, err := h.service.GetPrescriptionByID(id)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, "Prescription not found"))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Prescription retrieved successfully", prescription))
}

// -------------------- List Prescriptions By Medical Record --------------------
// @Summary List prescriptions for a medical record
// @Description Retrieve all prescriptions for a given medical record ID
// @Tags Prescriptions
// @Produce json
// @Param record_id path string true "Medical Record ID"
// @Success 200 {array} medicalrecord.Prescription
// @Failure 500 {object} map[string]string
// @Router /prescriptions/medical_records/{record_id} [get]
func (h *PrescriptionHandler) ListPrescriptionsByMedicalRecordID(c *gin.Context) {
	recordID := c.Param("record_id")
	prescriptions, err := h.service.GetPrescriptionsByMedicalRecordID(recordID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Prescriptions retrieved successfully", prescriptions))
}

// -------------------- List Prescriptions By Patient ID --------------------
// @Summary List prescriptions by patient ID with filters
// @Description Retrieve all prescriptions for a given patient ID with optional filters and sorting
// @Tags Prescriptions
// @Produce json
// @Param patient_id path string true "Patient ID"
// @Param status query string false "Filter by prescription status (PENDING/APPROVED/REJECTED)"
// @Param date query string false "Filter by creation date (format: YYYY-MM-DD)"
// @Param sort query string false "Sort by created date (newest/oldest, default: newest)"
// @Success 200 {array} medicalrecord.Prescription
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /prescriptions/patient/{patient_id} [get]
func (h *PrescriptionHandler) GetPrescriptionsByPatientID(c *gin.Context) {
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
		prescriptions, err := h.service.GetPrescriptionsByPatientIDWithFilters(patientID, status, date, sortBy)
		if err != nil {
			c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
			return
		}
		c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Prescriptions retrieved successfully", prescriptions))
		return
	}

	// Nếu không có filter thì dùng method cơ bản
	prescriptions, err := h.service.GetPrescriptionsByPatientID(patientID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Prescriptions retrieved successfully", prescriptions))
}

// -------------------- Approve Prescription --------------------
// @Summary Approve a prescription
// @Description Doctor approves a prescription
// @Tags Prescriptions
// @Produce json
// @Param prescription_id path string true "Prescription ID"
// @Param doctor_id query string true "Doctor ID"
// @Success 200 {object}  medicalrecord.Prescription
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /prescriptions/{prescription_id}/approve [put]
func (h *PrescriptionHandler) ApprovePrescription(c *gin.Context) {
	id := c.Param("prescription_id")
	doctorID := c.Query("doctor_id")

	if doctorID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Doctor ID is required"))
		return
	}

	if err := h.service.Approve(id, doctorID); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Prescription approved successfully", nil))
}

// -------------------- Update Prescription --------------------
// @Summary Update prescription
// @Description Update an existing prescription
// @Tags Prescriptions
// @Accept json
// @Produce json
// @Param prescription_id path string true "Prescription ID"
// @Param payload body updatePrescriptionReq true "Updated fields"
// @Success 200 {object} medicalrecord.Prescription
// @Failure 400 {object} map[string]string
// @Failure 404 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /prescriptions/{prescription_id} [put]
func (h *PrescriptionHandler) UpdatePrescription(c *gin.Context) {
	id := c.Param("prescription_id")
	var req updatePrescriptionReq

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	existing, err := h.service.GetPrescriptionByID(id)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, "Prescription not found"))
		return
	}

	if req.Status != "" {
		existing.Status = req.Status
	}
	existing.UpdatedAt = time.Now()

	if err := h.service.UpdatePrecription(existing); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Prescription updated successfully", existing))
}

// -------------------- Delete Prescription --------------------
// @Summary Delete prescription
// @Description Delete a prescription by ID
// @Tags Prescriptions
// @Produce json
// @Param prescription_id path string true "Prescription ID"
// @Success 200 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /prescriptions/{prescription_id} [delete]
func (h *PrescriptionHandler) DeletePrescription(c *gin.Context) {
	id := c.Param("prescription_id")

	if err := h.service.Delete(id); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Prescription deleted successfully", nil))
}

// -------------------- Get Medication Reminders By Patient ID --------------------
// @Summary Get medication reminders by patient ID for today
// @Description Retrieve all medication reminders for a given patient ID on today with prescription item details
// @Tags Prescriptions
// @Produce json
// @Param patient_id path string true "Patient ID"
// @Success 200 {array} medicalrecord.MedicationReminderWithItem
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /medication-reminders/patient/{patient_id} [get]
func (h *PrescriptionHandler) GetMedicationRemindersByPatientID(c *gin.Context) {
	patientID := c.Param("patient_id")

	if patientID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "patient_id is required"))
		return
	}

	reminders, err := h.service.GetMedicationRemindersByPatientID(patientID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Medication reminders retrieved successfully", reminders))
}
