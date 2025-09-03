package doctorhandler

import (
	"hospital-service/internal/config"
	"hospital-service/internal/enums"
	"hospital-service/internal/services/doctorservice"
	"net/http"

	"github.com/gin-gonic/gin"
)

type DoctorHandler struct {
	service *doctorservice.DoctorService
	cfg     config.Config
}

func NewDoctorHandler(cfg config.Config, service *doctorservice.DoctorService) *DoctorHandler {
	return &DoctorHandler{service: service, cfg: cfg}
}

// ----------- Request DTOs -----------

type createDoctorReq struct {
	UserID     string          `json:"user_id" binding:"required"`
	FullName   string          `json:"full_name" binding:"required"`
	Specialty  enums.Specialty `json:"specialty" binding:"required"`
	HospitalID string          `json:"hospital_id" binding:"required"`
	Phone      string          `json:"phone"`
	Email      string          `json:"email"`
}

type updateDoctorReq struct {
	FullName   string          `json:"full_name"`
	Specialty  enums.Specialty `json:"specialty"`
	HospitalID string          `json:"hospital_id"`
	Phone      string          `json:"phone"`
	Email      string          `json:"email"`
}

// ---------------- Create Doctor ----------------
// @Summary Create a new doctor
// @Description Add a new doctor record
// @Tags Doctors
// @Accept json
// @Produce json
// @Param doctor body createDoctorReq true "Doctor info"
// @Success 201 {object} doctor.Doctor
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /doctors [post]
func (h *DoctorHandler) CreateDoctor(c *gin.Context) {
	var req createDoctorReq
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	d, err := h.service.CreateDoctor(
		req.UserID,
		req.FullName,
		req.Specialty,
		req.HospitalID,
		req.Phone,
		req.Email,
	)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, d)
}

// ---------------- Get Doctor By ID ----------------
// @Summary Get doctor by doctor ID
// @Description Retrieve doctor record using doctor ID
// @Tags Doctors
// @Produce json
// @Param doctor_id path string true "Doctor ID"
// @Success 200 {object} doctor.Doctor
// @Failure 404 {object} map[string]string
// @Router /doctors/{doctor_id} [get]
func (h *DoctorHandler) GetDoctorByID(c *gin.Context) {
	doctorID := c.Param("doctor_id")
	d, err := h.service.GetDoctorByID(doctorID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "doctor not found"})
		return
	}
	c.JSON(http.StatusOK, d)
}


// ---------------- Get Doctor By UserID ----------------
// @Summary Get doctor by user ID
// @Description Retrieve doctor record using user ID
// @Tags Doctors
// @Produce json
// @Param user_id path string true "User ID"
// @Success 200 {object} doctor.Doctor
// @Failure 404 {object} map[string]string
// @Router /doctors/user/{user_id} [get]
func (h *DoctorHandler) GetDoctorByUserID(c *gin.Context) {
	userID := c.Param("user_id")
	p, err := h.service.FindByUserID(userID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "patient not found"})
		return
	}
	c.JSON(http.StatusOK, p)
}


// ---------------- Update Doctor ----------------
// @Summary Update doctor info
// @Description Update doctor record by doctor ID
// @Tags Doctors
// @Accept json
// @Produce json
// @Param doctor_id path string true "Doctor ID"
// @Param doctor body updateDoctorReq true "Updated doctor info"
// @Success 200 {object} doctor.Doctor
// @Failure 400 {object} map[string]string
// @Failure 404 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /doctors/{doctor_id} [put]
func (h *DoctorHandler) UpdateDoctor(c *gin.Context) {
	doctorID := c.Param("doctor_id")
	var req updateDoctorReq

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	d, err := h.service.GetDoctorByID(doctorID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "doctor not found"})
		return
	}

	// Chỉ update các trường có giá trị
	if req.FullName != "" {
		d.FullName = req.FullName
	}
	if req.Specialty != "" {
		d.Specialty = req.Specialty
	}
	if req.HospitalID != "" {
		d.HospitalID = req.HospitalID
	}
	if req.Phone != "" {
		d.Phone = req.Phone
	}
	if req.Email != "" {
		d.Email = req.Email
	}

	if err := h.service.UpdateDoctor(d); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, d)
}

// ---------------- Delete Doctor ----------------
// @Summary Delete doctor
// @Description Delete doctor by doctor ID
// @Tags Doctors
// @Produce json
// @Param doctor_id path string true "Doctor ID"
// @Success 200 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /doctors/{doctor_id} [delete]
func (h *DoctorHandler) DeleteDoctor(c *gin.Context) {
	doctorID := c.Param("doctor_id")
	if err := h.service.DeleteDoctor(doctorID); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"message": "doctor deleted"})
}

// ---------------- List Doctors ----------------
// @Summary List all doctors
// @Description Retrieve all doctors
// @Tags Doctors
// @Produce json
// @Success 200 {array} doctor.Doctor
// @Failure 500 {object} map[string]string
// @Router /doctors [get]
func (h *DoctorHandler) ListDoctors(c *gin.Context) {
	doctors, err := h.service.ListDoctors()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, doctors)
}

// ---------------- List Doctors by Hospital ID ----------------
// @Summary List doctors by hospital ID
// @Description Retrieve doctors for a specific hospital
// @Tags Doctors
// @Produce json
// @Param hospital_id path string true "Hospital ID"
// @Success 200 {array} doctor.Doctor
// @Failure 500 {object} map[string]string
// @Router /doctors/hospital/{hospital_id} [get]
func (h *DoctorHandler) ListDoctorsByHospitalID(c *gin.Context) {
	hospitalID := c.Param("hospital_id")

	doctors, err := h.service.FindByHospitalID(hospitalID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, doctors)
}
