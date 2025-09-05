package patienthandler

import (
	"hospital-service/internal/config"
	"hospital-service/internal/enums"
	"hospital-service/internal/services/patientservice"
	"hospital-service/internal/utils"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

type PatientHandler struct {
	service *patientservice.PatientService
	cfg     config.Config
}

func NewPatientHandler(cfg config.Config, service *patientservice.PatientService) *PatientHandler {
	return &PatientHandler{service: service, cfg: cfg}
}

type createPatientReq struct {
	UserID    string    `json:"user_id" binding:"required"`
	FullName  string    `json:"full_name" binding:"required"`
	DOB       time.Time `json:"dob" binding:"required"`
	Gender    string    `json:"gender" binding:"required,oneof=male female other"`
	Address   string    `json:"address"`
	Phone     string    `json:"phone"`
	Email     string    `json:"email"`
	AvatarURL string    `json:"avatar_url"`
}

type updatePatientReq struct {
	FullName string    `json:"full_name"`
	DOB      time.Time `json:"dob"`
	Gender   string    `json:"gender" binding:"oneof=male female other"`
	Address  string    `json:"address"`
	Phone    string    `json:"phone"`
	Email    string    `json:"email"`
}

// ---------------- Create Patient ----------------
// @Summary Create a new patient
// @Description Add a new patient record
// @Tags Patients
// @Accept json
// @Produce json
// @Param patient body createPatientReq true "Patient info"
// @Success 201 {object} patient.Patient
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /patients [post]
func (h *PatientHandler) CreatePatient(c *gin.Context) {
	var req createPatientReq
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	p, err := h.service.CreatePatient(req.UserID, req.FullName, req.DOB, enums.Gender(req.Gender), req.Address, req.Phone, req.Email, req.AvatarURL)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "Patient created successfully", p))
}

// ---------------- Get Patient By UserID ----------------
// @Summary Get patient by user ID
// @Description Retrieve patient record using user ID
// @Tags Patients
// @Produce json
// @Param user_id path string true "User ID"
// @Success 200 {object} patient.Patient
// @Failure 404 {object} map[string]string
// @Router /patients/user/{user_id} [get]
func (h *PatientHandler) GetPatientByUserID(c *gin.Context) {
	userID := c.Param("user_id")
	p, err := h.service.GetPatientByUserID(userID)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, "Patient not found"))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Patient retrieved successfully", p))
}

// ---------------- Get Patient By PatientID ----------------
// @Summary Get patient by patient ID
// @Description Retrieve patient record using patient ID
// @Tags Patients
// @Produce json
// @Param patient_id path string true "Patient ID"
// @Success 200 {object} patient.Patient
// @Failure 404 {object} map[string]string
// @Router /patients/{patient_id} [get]
func (h *PatientHandler) GetPatientByID(c *gin.Context) {
	patientID := c.Param("patient_id")
	p, err := h.service.GetPatientByID(patientID)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, "Patient not found"))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Patient retrieved successfully", p))
}

// ---------------- Update Patient ----------------
// @Summary Update patient info
// @Description Update patient record by patient ID
// @Tags Patients
// @Accept json
// @Produce json
// @Param patient_id path string true "Patient ID"
// @Param patient body updatePatientReq true "Updated patient info"
// @Success 200 {object} patient.Patient
// @Failure 400 {object} map[string]string
// @Failure 404 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /patients/{patient_id} [put]
func (h *PatientHandler) UpdatePatient(c *gin.Context) {
	patientID := c.Param("patient_id")
	var req updatePatientReq
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	p, err := h.service.GetPatientByID(patientID)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, "Patient not found"))
		return
	}

	if req.FullName != "" {
		p.FullName = req.FullName
	}
	if !req.DOB.IsZero() {
		p.DOB = req.DOB
	}
	if req.Gender != "" {
		p.Gender = enums.Gender(req.Gender)
	}
	if req.Address != "" {
		p.Address = req.Address
	}
	if req.Phone != "" {
		p.Phone = req.Phone
	}
	if req.Email != "" {
		p.Email = req.Email
	}

	if err := h.service.UpdatePatient(p); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Patient updated successfully", p))
}

// ---------------- Delete Patient ----------------
// @Summary Delete patient
// @Description Delete patient by patient ID
// @Tags Patients
// @Produce json
// @Param patient_id path string true "Patient ID"
// @Success 200 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /patients/{patient_id} [delete]
func (h *PatientHandler) DeletePatient(c *gin.Context) {
	patientID := c.Param("patient_id")
	if err := h.service.DeletePatient(patientID); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Patient deleted successfully", nil))
}

// ---------------- List Patients ----------------
// @Summary List all patients
// @Description Retrieve all patients
// @Tags Patients
// @Produce json
// @Success 200 {array} patient.Patient
// @Failure 500 {object} map[string]string
// @Router /patients [get]
func (h *PatientHandler) ListPatients(c *gin.Context) {
	patients, err := h.service.ListPatients()
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Patients retrieved successfully", patients))
}
