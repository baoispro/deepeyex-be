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
	PatientID     string `json:"patient_id" binding:"required"`
	DoctorID      string `json:"doctor_id" binding:"required"`
	AppointmentID string `json:"appointment_id,omitempty"` // Optional
	Diagnosis     string `json:"diagnosis" binding:"required"`
	CreatedBy     string `json:"created_by" binding:"required"` // "DOCTOR" | "SYSTEM" | "AI"
}

type CreateAIDiagnosisRequest struct {
	DiseaseCode string  `json:"disease_code" binding:"required"`
	Confidence  float64 `json:"confidence" binding:"required"`
}


func NewMedicalRecordHandler(cfg config.Config ,service *medicalrecordservice.MedicalRecordService) *MedicalRecordHandler {
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

	record, err := h.service.CreateRecord(req.PatientID, req.DoctorID, req.Diagnosis, req.CreatedBy, req.AppointmentID)
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

// @Summary Add AI Diagnosis to Medical Record
// @Description Add a new AI diagnosis to a specific medical record
// @Tags AI_Diagnoses
// @Accept json
// @Produce json
// @Param record_id path string true "Medical Record ID"
// @Param payload body CreateAIDiagnosisRequest true "AI Diagnosis payload"
// @Success 201 {object} medicalrecord.AIDiagnosis
// @Failure 400 {object} map[string]interface{}
// @Failure 500 {object} map[string]interface{}
// @Router /medical_records/{record_id}/ai_diagnoses [post]
func (h *MedicalRecordHandler) AddAIDiagnosis(c *gin.Context) {
	recordID := c.Param("record_id")

	var req CreateAIDiagnosisRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	diagnosis, err := h.service.AddAIDiagnosisByRecordID(recordID, req.DiseaseCode, req.Confidence)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "AI Diagnosis created successfully", diagnosis))
}

// @Summary List AI Diagnoses by Record ID
// @Description Get a list of AI diagnoses for a specific medical record
// @Tags AI_Diagnoses
// @Accept json
// @Produce json
// @Param record_id path string true "Medical Record ID"
// @Success 200 {array} medicalrecord.AIDiagnosis
// @Failure 400 {object} map[string]interface{}
// @Failure 500 {object} map[string]interface{}
// @Router /medical_records/{record_id}/ai_diagnoses [get]
func (h *MedicalRecordHandler) ListAIDiagnoses(c *gin.Context) {
	recordID := c.Param("record_id")

	if recordID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "record_id is required"))
		return
	}

	diagnoses, err := h.service.ListAIDiagnoses(recordID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "AI Diagnoses retrieved successfully", diagnoses))
}

// @Summary Get AI Diagnosis by ID
// @Description Get the details of a specific AI diagnosis by its ID
// @Tags AI_Diagnoses
// @Accept json
// @Produce json
// @Param id path string true "AI Diagnosis ID"
// @Success 200 {object} medicalrecord.AIDiagnosis
// @Failure 400 {object} map[string]interface{}
// @Failure 404 {object} map[string]interface{}
// @Failure 500 {object} map[string]interface{}
// @Router /medical_records/ai_diagnoses/{id} [get]
func (h *MedicalRecordHandler) GetAIDiagnosisByID(c *gin.Context) {
	id := c.Param("id")

	if id == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "diagnosis_id is required"))
		return
	}

	diagnosis, err := h.service.GetAIDiagnosisByID(id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	if diagnosis == nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, "AI Diagnosis not found"))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "AI Diagnosis retrieved successfully", diagnosis))
}


// ---------------- Delete AI Diagnosis ----------------
// @Summary Delete AI Diagnosis by ID
// @Description Delete a specific AI diagnosis by its ID
// @Tags AI_Diagnoses
// @Accept json
// @Produce json
// @Param id path string true "AI Diagnosis ID"
// @Success 200 {object} map[string]string
// @Failure 400 {object} map[string]interface{}
// @Failure 500 {object} map[string]interface{}
// @Router /medical_records/ai_diagnoses/{id} [delete]
func (h *MedicalRecordHandler) DeleteAIDiagnosis(c *gin.Context) {
	id := c.Param("id")

	if id == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "diagnosis_id is required"))
		return
	}

	if err := h.service.DeleteAIDiagnosis(id); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "AI Diagnosis deleted successfully", nil))
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

// @Summary List AI Recommended Plans
// @Description Get a list of recommended treatment plans for a specific AI Diagnosis
// @Tags AI_RecommendedPlans
// @Accept json
// @Produce json
// @Param diagnosis_id path string true "AI Diagnosis ID"
// @Success 200 {array} medicalrecord.AIRecommendedPlan
// @Failure 400 {object} map[string]interface{}
// @Failure 500 {object} map[string]interface{}
// @Router /medical_records/ai_diagnoses/{diagnosis_id}/recommended_plans [get]
func (h *MedicalRecordHandler) ListRecommendedPlans(c *gin.Context) {
	diagnosisID := c.Param("diagnosis_id")

	if diagnosisID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "diagnosis_id is required"))
		return
	}

	plans, err := h.service.ListRecommendedPlans(diagnosisID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Recommended plans retrieved successfully", plans))
}

// @Summary Add a Recommended Plan
// @Description Add a new recommended treatment plan to an AI Diagnosis
// @Tags AI_RecommendedPlans
// @Accept json
// @Produce json
// @Param diagnosis_id path string true "AI Diagnosis ID"
// @Param payload body medicalrecord.AddRecommendedPlanRequest true "Recommended Plan payload"
// @Success 201 {object} medicalrecord.AIRecommendedPlan
// @Failure 400 {object} map[string]interface{}
// @Failure 500 {object} map[string]interface{}
// @Router /medical_records/ai_diagnoses/{diagnosis_id}/recommended_plans [post]
func (h *MedicalRecordHandler) AddRecommendedPlan(c *gin.Context) {
	diagnosisID := c.Param("diagnosis_id")

	if diagnosisID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "diagnosis_id is required"))
		return
	}

	var req medicalrecord.AddRecommendedPlanRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	plan, err := h.service.AddRecommendedPlan(
		diagnosisID,
		req.Description,
		req.DrugName,
		req.Dosage,
		req.Frequency,
		req.DurationDays,
	)

	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "Recommended plan created successfully", plan))
}

// @Summary Delete a Recommended Plan
// @Description Delete a recommended treatment plan by its ID
// @Tags AI_RecommendedPlans
// @Accept json
// @Produce json
// @Param plan_id path string true "Recommended Plan ID"
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]interface{}
// @Failure 404 {object} map[string]interface{}
// @Failure 500 {object} map[string]interface{}
// @Router /medical_records/ai_recommended_plans/{plan_id} [delete]
func (h *MedicalRecordHandler) DeleteRecommendedPlan(c *gin.Context) {
	planID := c.Param("plan_id")

	if planID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "plan_id is required"))
		return
	}

	if err := h.service.DeleteRecommendedPlan(planID); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Recommended plan deleted successfully", nil))
}
