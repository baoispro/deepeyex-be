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

// ---------------- Create Patient ----------------
// @Summary Create a new patient
// @Description Add a new patient record with optional avatar upload
// @Tags Patients
// @Accept multipart/form-data
// @Produce json
// @Param user_id formData string true "User ID"
// @Param full_name formData string true "Full Name"
// @Param dob formData string true "Date of birth (YYYY-MM-DD)"
// @Param gender formData string true "Gender (male/female/other)"
// @Param address formData string false "Address"
// @Param phone formData string false "Phone"
// @Param email formData string false "Email"
// @Param avatar formData file false "Avatar file"
// @Success 201 {object} patient.Patient
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /patients [post]
func (h *PatientHandler) CreatePatient(c *gin.Context) {
	userID := c.PostForm("user_id")
	fullName := c.PostForm("full_name")
	dobStr := c.PostForm("dob")
	gender := c.PostForm("gender")
	address := c.PostForm("address")
	phone := c.PostForm("phone")
	email := c.PostForm("email")

	// parse dob
	dob, err := time.Parse("2006-01-02", dobStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Invalid date format, must be YYYY-MM-DD"))
		return
	}

	// lấy avatar file (nếu có)
	var avatarFile interface{}
	fileHeader, err := c.FormFile("avatar")
	if err == nil { // có file
		avatarFile = fileHeader
	}

	p, err := h.service.CreatePatient(userID, fullName, dob, enums.Gender(gender),
		address, phone, email, avatarFile)
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
// @Description Update patient record by patient ID (with optional new avatar)
// @Tags Patients
// @Accept multipart/form-data
// @Produce json
// @Param patient_id path string true "Patient ID"
// @Param full_name formData string false "Full Name"
// @Param dob formData string false "Date of birth (YYYY-MM-DD)"
// @Param gender formData string false "Gender (male/female/other)"
// @Param address formData string false "Address"
// @Param phone formData string false "Phone"
// @Param email formData string false "Email"
// @Param avatar formData file false "Avatar file"
// @Success 200 {object} patient.Patient
// @Failure 400 {object} map[string]string
// @Failure 404 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /patients/{patient_id} [put]
func (h *PatientHandler) UpdatePatient(c *gin.Context) {
	patientID := c.Param("patient_id")

	p, err := h.service.GetPatientByID(patientID)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, "Patient not found"))
		return
	}

	if fullName := c.PostForm("full_name"); fullName != "" {
		p.FullName = fullName
	}
	if dobStr := c.PostForm("dob"); dobStr != "" {
		dob, err := time.Parse("2006-01-02", dobStr)
		if err != nil {
			c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Invalid date format"))
			return
		}
		p.DOB = dob
	}
	if gender := c.PostForm("gender"); gender != "" {
		p.Gender = enums.Gender(gender)
	}
	if address := c.PostForm("address"); address != "" {
		p.Address = address
	}
	if phone := c.PostForm("phone"); phone != "" {
		p.Phone = phone
	}
	if email := c.PostForm("email"); email != "" {
		p.Email = email
	}

	// lấy avatar file (nếu có)
	var avatarFile interface{}
	fileHeader, err := c.FormFile("avatar")
	if err == nil {
		avatarFile = fileHeader
	}

	if err := h.service.UpdatePatient(p, avatarFile); err != nil {
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
