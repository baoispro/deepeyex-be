package doctorhandler

import (
	"hospital-service/internal/config"
	"hospital-service/internal/enums"
	"hospital-service/internal/services/doctorservice"
	"hospital-service/internal/utils"
	"mime/multipart"
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
	UserID     string          `form:"user_id" binding:"required"`
	FullName   string          `form:"full_name" binding:"required"`
	Specialty  enums.Specialty `form:"specialty" binding:"required"`
	HospitalID string          `form:"hospital_id" binding:"required"`
	Phone      string          `form:"phone"`
	Email      string          `form:"email"`
	// AvatarFile nhận file upload
	AvatarFile *multipart.FileHeader `form:"avatar"`
}

type updateDoctorReq struct {
	FullName   string                `form:"full_name"`
	Specialty  enums.Specialty       `form:"specialty"`
	HospitalID string                `form:"hospital_id"`
	Phone      string                `form:"phone"`
	Email      string                `form:"email"`
	AvatarFile *multipart.FileHeader `form:"avatar"`
}

// ---------------- Create Doctor ----------------
// @Summary Create a new doctor
// @Description Add a doctor with required info and optional avatar upload
// @Tags Doctors
// @Accept multipart/form-data
// @Produce json
// @Param user_id formData string true "User ID"
// @Param full_name formData string true "Full Name"
// @Param specialty formData string true "Specialty"
// @Param hospital_id formData string true "Hospital ID"
// @Param phone formData string false "Phone"
// @Param email formData string false "Email"
// @Param avatar formData file false "Avatar File"
// @Success 201 {object} utils.APIResponse
// @Failure 400 {object} utils.APIResponse
// @Failure 500 {object} utils.APIResponse
// @Router /doctors [post]
func (h *DoctorHandler) CreateDoctor(c *gin.Context) {
	var req createDoctorReq
	if err := c.ShouldBind(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	var avatarFile interface{}
	if req.AvatarFile != nil {
		avatarFile = req.AvatarFile // ép thẳng
	}

	d, err := h.service.CreateDoctor(
		req.UserID,
		req.FullName,
		req.Specialty,
		req.HospitalID,
		req.Phone,
		req.Email,
		avatarFile,
	)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "Doctor created successfully", d))
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
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, "Doctor not found"))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Doctor retrieved successfully", d))
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
	d, err := h.service.FindByUserID(userID)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, "Doctor not found"))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Doctor retrieved successfully", d))
}

// ---------------- Update Doctor ----------------
// @Summary Update a doctor
// @Description Update doctor info and avatar
// @Tags Doctors
// @Accept multipart/form-data
// @Produce json
// @Param doctor_id path string true "Doctor ID"
// @Param full_name formData string false "Full Name"
// @Param specialty formData string false "Specialty"
// @Param hospital_id formData string false "Hospital ID"
// @Param phone formData string false "Phone"
// @Param email formData string false "Email"
// @Param avatar formData file false "Avatar File"
// @Success 200 {object} utils.APIResponse
// @Failure 400 {object} utils.APIResponse
// @Failure 404 {object} utils.APIResponse
// @Failure 500 {object} utils.APIResponse
// @Router /doctors/{doctor_id} [put]
func (h *DoctorHandler) UpdateDoctor(c *gin.Context) {
	doctorID := c.Param("doctor_id")
	var req updateDoctorReq

	if err := c.ShouldBind(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	d, err := h.service.GetDoctorByID(doctorID)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, "Doctor not found"))
		return
	}

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

	var avatarFile interface{}
	if req.AvatarFile != nil {
		avatarFile = req.AvatarFile
	}

	if err := h.service.UpdateDoctor(d, avatarFile); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Doctor updated successfully", d))
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
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Doctor deleted successfully", nil))
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
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Doctors retrieved successfully", doctors))
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
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Doctors retrieved successfully", doctors))
}
