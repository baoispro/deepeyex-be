package fullrecordhandler

import (
	"encoding/json"
	"hospital-service/internal/services/fullrecordservice"
	"hospital-service/internal/services/medicalrecordservice"
	"hospital-service/internal/utils"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
)

type FullRecordHandler struct {
	service *fullrecordservice.FullRecordService
}

func NewFullRecordHandler(service *fullrecordservice.FullRecordService) *FullRecordHandler {
	return &FullRecordHandler{service: service}
}

// ---------------- Create Full Record ----------------
// @Summary Create full medical record
// @Description Create a new medical record with attachments and prescription
// @Tags Medical Records
// @Accept multipart/form-data
// @Produce json
// @Param patient_id formData string true "Patient ID"
// @Param doctor_id formData string true "Doctor ID"
// @Param diagnosis formData string true "Diagnosis"
// @Param appointment_id formData string false "Appointment ID"
// @Param notes formData string false "Notes"
// @Param related_record_id formData string false "Related Record ID"
// @Param prescription formData string false "Prescription JSON string"
// @Param files formData file false "Attachment files (multiple)"
// @Param file_types formData string false "File types (comma-separated, e.g. X-RAY,LAB_RESULT)"
// @Success 201 {object} medicalrecord.MedicalRecord
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /full-records/full [post]
func (h *FullRecordHandler) CreateFullRecord(c *gin.Context) {
	// Parse form data
	patientID := c.PostForm("patient_id")
	doctorID := c.PostForm("doctor_id")
	diagnosis := c.PostForm("diagnosis")
	appointmentID := c.PostForm("appointment_id")
	notes := c.PostForm("notes")
	relatedRecordID := c.PostForm("related_record_id")
	prescriptionJSON := c.PostForm("prescription")
	fileTypesStr := c.PostForm("file_types")

	// Validate required fields
	if patientID == "" || doctorID == "" || diagnosis == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "patient_id, doctor_id, and diagnosis are required"))
		return
	}

	// Build request
	req := fullrecordservice.FullRecordCreateRequest{
		PatientID:     patientID,
		DoctorID:      doctorID,
		Diagnosis:     diagnosis,
		AppointmentID: appointmentID,
	}

	if notes != "" {
		req.Notes = &notes
	}
	if relatedRecordID != "" {
		req.RelatedRecordID = &relatedRecordID
	}

	// Parse prescription if provided
	if prescriptionJSON != "" {
		var prescription medicalrecordservice.CreatePrescriptionRequest
		if err := json.Unmarshal([]byte(prescriptionJSON), &prescription); err != nil {
			c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Invalid prescription JSON: "+err.Error()))
			return
		}
		req.Prescription = &prescription
	}

	// Parse attachments
	form, err := c.MultipartForm()
	if err == nil && form != nil && form.File["files"] != nil {
		files := form.File["files"]
		
		// Parse file types (comma-separated)
		fileTypes := []string{}
		if fileTypesStr != "" {
			// Split by comma
			for _, ft := range strings.Split(fileTypesStr, ",") {
				fileTypes = append(fileTypes, strings.TrimSpace(ft))
			}
		}

		// Create attachment requests
		for i, fileHeader := range files {
			fileType := "OTHER"
			if i < len(fileTypes) {
				fileType = fileTypes[i]
			}

			req.Attachments = append(req.Attachments, fullrecordservice.AttachmentRequest{
				FileType: fileType,
				File:     fileHeader,
			})
		}
	}

	record, err := h.service.CreateFullRecord(&req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "Full medical record created successfully", record))
}

// ---------------- Complete Record ----------------
// @Summary Complete existing medical record
// @Description Update an existing record with diagnosis, notes, attachments, and prescription
// @Tags Medical Records
// @Accept multipart/form-data
// @Produce json
// @Param record_id formData string true "Record ID"
// @Param diagnosis formData string true "Diagnosis"
// @Param notes formData string false "Notes"
// @Param update_doctor formData string false "Update Doctor ID"
// @Param update_patient formData string false "Update Patient ID"
// @Param prescription formData string false "Prescription JSON string"
// @Param files formData file false "Attachment files (multiple)"
// @Param file_types formData string false "File types (comma-separated)"
// @Success 200 {object} medicalrecord.MedicalRecord
// @Failure 400 {object} map[string]string
// @Failure 404 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /full-records/complete [put]
func (h *FullRecordHandler) CompleteRecord(c *gin.Context) {
	// Parse form data
	recordID := c.PostForm("record_id")
	diagnosis := c.PostForm("diagnosis")
	notes := c.PostForm("notes")
	updateDoctor := c.PostForm("update_doctor")
	updatePatient := c.PostForm("update_patient")
	prescriptionJSON := c.PostForm("prescription")
	fileTypesStr := c.PostForm("file_types")

	// Validate required fields
	if recordID == "" || diagnosis == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "record_id and diagnosis are required"))
		return
	}

	// Build request
	req := fullrecordservice.CompleteRecordRequest{
		RecordID:  recordID,
		Diagnosis: diagnosis,
	}

	if notes != "" {
		req.Notes = &notes
	}
	if updateDoctor != "" {
		req.UpdateDoctor = &updateDoctor
	}
	if updatePatient != "" {
		req.UpdatePatient = &updatePatient
	}

	// Parse prescription if provided
	if prescriptionJSON != "" {
		var prescription medicalrecordservice.CreatePrescriptionRequest
		if err := json.Unmarshal([]byte(prescriptionJSON), &prescription); err != nil {
			c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Invalid prescription JSON: "+err.Error()))
			return
		}
		req.Prescription = &prescription
	}

	// Parse attachments
	form, err := c.MultipartForm()
	if err == nil && form != nil && form.File["files"] != nil {
		files := form.File["files"]
		
		// Parse file types (comma-separated)
		fileTypes := []string{}
		if fileTypesStr != "" {
			for _, ft := range strings.Split(fileTypesStr, ",") {
				fileTypes = append(fileTypes, strings.TrimSpace(ft))
			}
		}

		// Create attachment requests
		for i, fileHeader := range files {
			fileType := "OTHER"
			if i < len(fileTypes) {
				fileType = fileTypes[i]
			}

			req.Attachments = append(req.Attachments, fullrecordservice.AttachmentRequest{
				FileType: fileType,
				File:     fileHeader,
			})
		}
	}

	record, err := h.service.CompleteRecord(&req)
	if err != nil {
		// kiểm tra lỗi not found hoặc khác
		if err.Error() == "medical record not found" {
			c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, err.Error()))
		} else {
			c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		}
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Medical record completed successfully", record))
}
