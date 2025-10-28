package medicalrecordhandler

import (
	"net/http"

	"hospital-service/internal/config"
	"hospital-service/internal/models/medicalrecord"
	"hospital-service/internal/services/medicalrecordservice"
	"hospital-service/internal/utils"

	"github.com/gin-gonic/gin"
)

type MedicalRecordHandler struct {
	service *medicalrecordservice.MedicalRecordService
	cfg     config.Config
}

type CreateMedicalRecordRequest struct {
	PatientID       string  `json:"patient_id" binding:"required"`
	DoctorID        string  `json:"doctor_id" binding:"required"`
	AppointmentID   string  `json:"appointment_id,omitempty"` // Optional
	Diagnosis       string  `json:"diagnosis" binding:"required"`
	CreatedBy       string  `json:"created_by" binding:"required"` // "DOCTOR" | "SYSTEM" | "AI"
	Note            *string `json:"note,omitempty"`
	RelatedRecordID *string `json:"related_record_id,omitempty"`
}

type CreateAIDiagnosisRequest struct {
	DiseaseCode  string  `json:"disease_code" binding:"required"`
	Confidence   float64 `json:"confidence" binding:"required"`
	MainImageURL string  `json:"main_image_url" binding:"required"` // URL hình ảnh chẩn đoán
	EyeType      *string `json:"eye_type,omitempty"`                // "LEFT" | "RIGHT" | "BOTH"
	Notes        *string `json:"notes,omitempty"`                   // Ghi chú
}

func NewMedicalRecordHandler(cfg config.Config, service *medicalrecordservice.MedicalRecordService) *MedicalRecordHandler {
	return &MedicalRecordHandler{service: service, cfg: cfg}
}

// ---------------- Create MedicalRecord ----------------
// @Summary Create a new medical record
// @Description Create a medical record for a patient with optional appointment and doctor.
// @Tags MedicalRecords
// @Accept json
// @Produce json
// @Param payload body CreateMedicalRecordRequest true "Medical Record Payload"
// @Success 201 {object} medicalrecord.MedicalRecord
// @Failure 400 {object} map[string]interface{}
// @Failure 500 {object} map[string]interface{}
// @Router /medical_records [post]
func (h *MedicalRecordHandler) CreateMedicalRecord(c *gin.Context) {
	var req CreateMedicalRecordRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Invalid request payload: "+err.Error()))
		return
	}

	record, err := h.service.CreateRecord(req.PatientID, req.DoctorID, req.Diagnosis, req.AppointmentID, req.Note, req.RelatedRecordID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "Medical record created successfully", record))
}

// ---------------- Get MedicalRecord by ID ----------------
// @Summary Get medical record by ID
// @Description Retrieve a medical record by record ID, including AI diagnoses and recommended plans
// @Tags MedicalRecords
// @Produce json
// @Param record_id path string true "MedicalRecord ID"
// @Success 200 {object} medicalrecord.MedicalRecord
// @Failure 404 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /medical_records/{record_id} [get]
func (h *MedicalRecordHandler) GetMedicalRecord(c *gin.Context) {
	id := c.Param("record_id")
	record, err := h.service.GetRecord(id)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, "Medical record not found"))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Medical record retrieved successfully", record))
}

// ---------------- List MedicalRecords ----------------
// @Summary List all medical records
// @Description Retrieve all medical records (optionally filtered by patient or doctor)
// @Tags MedicalRecords
// @Produce json
// @Success 200 {array} medicalrecord.MedicalRecord
// @Failure 500 {object} map[string]string
// @Router /medical_records [get]
func (h *MedicalRecordHandler) ListMedicalRecords(c *gin.Context) {
	list, err := h.service.ListRecords()
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Medical records retrieved successfully", list))
}

// ---------------- Update MedicalRecord ----------------
// @Summary Update medical record
// @Description Update an existing medical record, including diagnosis and recommended plan
// @Tags MedicalRecords
// @Accept json
// @Produce json
// @Param record_id path string true "MedicalRecord ID"
// @Param medical_record body medicalrecord.MedicalRecord true "Updated MedicalRecord payload"
// @Success 200 {object} medicalrecord.MedicalRecord
// @Failure 400 {object} map[string]string
// @Failure 404 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /medical_records/{record_id} [put]
func (h *MedicalRecordHandler) UpdateMedicalRecord(c *gin.Context) {
	id := c.Param("record_id")
	record, err := h.service.GetRecord(id)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, "Medical record not found"))
		return
	}

	var req medicalrecord.MedicalRecord
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	// Simple update: overwrite fields
	record.Diagnosis = req.Diagnosis
	record.DoctorID = req.DoctorID
	record.UpdatedAt = req.UpdatedAt

	if err := h.service.UpdateRecord(record); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Medical record updated successfully", record))
}

// ---------------- Delete MedicalRecord ----------------
// @Summary Delete medical record
// @Description Delete a medical record by ID
// @Tags MedicalRecords
// @Produce json
// @Param record_id path string true "MedicalRecord ID"
// @Success 200 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /medical_records/{record_id} [delete]
func (h *MedicalRecordHandler) DeleteMedicalRecord(c *gin.Context) {
	id := c.Param("record_id")
	if err := h.service.DeleteRecord(id); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Medical record deleted successfully", nil))
}

// ---------------- Init MedicalRecord and AI Diagnosis ----------------
// @Summary Init Medical Record and AI Diagnosis
// @Description Create a new medical record and attach an AI diagnosis in a single request
// @Tags MedicalRecords
// @Accept json
// @Produce json
// @Param payload body medicalrecord.InitRecordAndDiagnosisRequest true "Medical record and AI Diagnosis payload"
// @Success 201 {object} medicalrecord.InitRecordAndDiagnosisResponse
// @Failure 400 {object} map[string]interface{} "Bad Request"
// @Failure 500 {object} map[string]interface{} "Internal Server Error"
// @Router /medical_records/init [post]
func (h *MedicalRecordHandler) InitMedicalRecordAndDiagnosis(c *gin.Context) {
	var req medicalrecord.InitRecordAndDiagnosisRequest

	// Bind JSON
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	// Call service
	result, err := h.service.InitRecordAndDiagnosis(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	// Success
	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "Medical record and AI diagnosis created successfully", result))
}

// ---------------- Check or Prepare MedicalRecord ----------------
// @Summary Check if a medical record exists for an appointment
// @Description Check whether a MedicalRecord exists for the given appointment.
//
//	Returns existing record for update or empty data for create form.
//
// @Tags MedicalRecords
// @Produce json
// @Param appointment_id query string true "Appointment ID"
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]interface{} "Bad Request"
// @Failure 500 {object} map[string]interface{} "Internal Server Error"
// @Router /medical_records/check [get]
func (h *MedicalRecordHandler) CheckMedicalRecord(c *gin.Context) {
	appointmentID := c.Query("appointment_id")
	if appointmentID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "appointment_id is required"))
		return
	}

	record, isUpdate, err := h.service.CheckRecordByAppointment(appointmentID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, "Không thể kiểm tra MedicalRecord: "+err.Error()))
		return
	}

	if isUpdate {
		// Trả về record hiện có để frontend hiển thị update form
		c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "MedicalRecord tồn tại, chuẩn bị update", gin.H{
			"action": "update",
			"record": record,
		}))
		return
	}

	// Trả về empty data để frontend hiển thị create form
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Chưa có MedicalRecord, chuẩn bị tạo mới", gin.H{
		"action": "create",
		"record": []interface{}{},
	}))
}

// ---------------- Get all MedicalRecords by PatientID ----------------
// @Summary Get all medical records for a patient
// @Description Returns a list of MedicalRecords along with related Attachments, Prescriptions, AI Diagnoses, and Appointment info.
// @Tags MedicalRecords
// @Produce json
// @Param patient_id query string true "Patient ID"
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]interface{} "Bad Request"
// @Failure 500 {object} map[string]interface{} "Internal Server Error"
// @Router /medical_records/patient [get]
func (h *MedicalRecordHandler) GetRecordsByPatient(c *gin.Context) {
	patientID := c.Query("patient_id")
	
	if patientID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "patient_id is required"))
		return
	}

	records, err := h.service.GetRecordsByPatient(patientID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, "Không thể lấy MedicalRecords: "+err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Lấy danh sách MedicalRecords thành công", gin.H{
		"records": records,
	}))
}

// ---------------- Get all MedicalRecords by PatientID ----------------
// @Summary Get all medical records for a patient
// @Description Returns a list of MedicalRecords along with related Attachments, Prescriptions, AI Diagnoses, and Appointment info.
// @Tags MedicalRecords
// @Produce json
// @Param patient_id query string true "Patient ID"
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]interface{} "Bad Request"
// @Failure 500 {object} map[string]interface{} "Internal Server Error"
// @Router /medical_records/patient_fe [get]
func (h *MedicalRecordHandler) GetRecordsByPatientFe(c *gin.Context) {
	patientID := c.Query("patient_id")
	
	if patientID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "patient_id is required"))
		return
	}

	records, err := h.service.GetRecordsByPatientFe(patientID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, "Không thể lấy MedicalRecords: "+err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Lấy danh sách MedicalRecords thành công", gin.H{
		"records": records,
	}))
}