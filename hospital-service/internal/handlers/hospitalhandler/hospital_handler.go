package hospitalhandler

import (
	"hospital-service/internal/config"
	"hospital-service/internal/services/hospitalservice"
	"hospital-service/internal/utils"
	"mime/multipart"
	"net/http"

	"github.com/gin-gonic/gin"
)

// HospitalHandler quản lý API endpoint cho Hospital
type HospitalHandler struct {
	service *hospitalservice.HospitalService
	cfg     config.Config
}

// NewHospitalHandler khởi tạo handler mới
func NewHospitalHandler(cfg config.Config, service *hospitalservice.HospitalService) *HospitalHandler {
	return &HospitalHandler{service: service, cfg: cfg}
}

// ----------- Request Structs -----------

type createHospitalReq struct {
	Name    string                `form:"name" binding:"required"`
	Address string                `form:"address"`
	Phone   string                `form:"phone"`
	Email   string                `form:"email"`
	Logo    *multipart.FileHeader `form:"logo"`
}

type updateHospitalReq struct {
	Name    string                `form:"name"`
	Address string                `form:"address"`
	Phone   string                `form:"phone"`
	Email   string                `form:"email"`
	Logo    *multipart.FileHeader `form:"logo"`
}

// ---------------- Create Hospital ----------------
// @Summary Create a new hospital
// @Description Add hospital info with optional logo upload
// @Tags Hospitals
// @Accept multipart/form-data
// @Produce json
// @Param name formData string true "Hospital Name"
// @Param address formData string false "Address"
// @Param phone formData string false "Phone"
// @Param email formData string false "Email"
// @Param logo formData file false "Hospital Logo"
// @Success 201 {object} utils.APIResponse
// @Failure 400 {object} utils.APIResponse
// @Failure 500 {object} utils.APIResponse
// @Router /hospitals [post]
func (h *HospitalHandler) CreateHospital(c *gin.Context) {
	var req createHospitalReq
	if err := c.ShouldBind(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	var logoFile interface{}
	if req.Logo != nil {
		logoFile = req.Logo
	}

	hospital, err := h.service.CreateHospital(req.Name, req.Address, req.Phone, req.Email, logoFile)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "Hospital created successfully", hospital))
}

// ---------------- Get Hospital By ID ----------------
// @Summary Get hospital by ID
// @Description Retrieve hospital record using hospital ID
// @Tags Hospitals
// @Produce json
// @Param hospital_id path string true "Hospital ID"
// @Success 200 {object} hospital.Hospital
// @Failure 404 {object} map[string]string
// @Router /hospitals/{hospital_id} [get]
func (h *HospitalHandler) GetHospitalByID(c *gin.Context) {
	hospitalID := c.Param("hospital_id")
	hospitalData, err := h.service.GetHospitalByID(hospitalID)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, "Hospital not found"))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Hospital retrieved successfully", hospitalData))
}

// ---------------- Update Hospital ----------------
// @Summary Update a hospital
// @Description Update hospital info and logo
// @Tags Hospitals
// @Accept multipart/form-data
// @Produce json
// @Param hospital_id path string true "Hospital ID"
// @Param name formData string false "Hospital Name"
// @Param address formData string false "Address"
// @Param phone formData string false "Phone"
// @Param email formData string false "Email"
// @Param logo formData file false "Hospital Logo"
// @Success 200 {object} utils.APIResponse
// @Failure 400 {object} utils.APIResponse
// @Failure 404 {object} utils.APIResponse
// @Failure 500 {object} utils.APIResponse
// @Router /hospitals/{hospital_id} [put]
func (h *HospitalHandler) UpdateHospital(c *gin.Context) {
	hospitalID := c.Param("hospital_id")
	var req updateHospitalReq

	if err := c.ShouldBind(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	hospitalData, err := h.service.GetHospitalByID(hospitalID)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, "Hospital not found"))
		return
	}

	if req.Name != "" {
		hospitalData.Name = req.Name
	}
	if req.Address != "" {
		hospitalData.Address = req.Address
	}
	if req.Phone != "" {
		hospitalData.Phone = req.Phone
	}
	if req.Email != "" {
		hospitalData.Email = req.Email
	}

	var logoFile interface{}
	if req.Logo != nil {
		logoFile = req.Logo
	}

	if err := h.service.UpdateHospital(hospitalData, logoFile); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Hospital updated successfully", hospitalData))
}

// ---------------- Delete Hospital ----------------
// @Summary Delete hospital
// @Description Delete hospital by hospital ID
// @Tags Hospitals
// @Produce json
// @Param hospital_id path string true "Hospital ID"
// @Success 200 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /hospitals/{hospital_id} [delete]
func (h *HospitalHandler) DeleteHospital(c *gin.Context) {
	hospitalID := c.Param("hospital_id")

	if err := h.service.DeleteHospital(hospitalID); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Hospital deleted successfully", nil))
}

// ---------------- List Hospitals ----------------
// @Summary List all hospitals
// @Description Retrieve all hospitals
// @Tags Hospitals
// @Produce json
// @Success 200 {array} hospital.Hospital
// @Failure 500 {object} map[string]string
// @Router /hospitals [get]
func (h *HospitalHandler) ListHospitals(c *gin.Context) {
	hospitals, err := h.service.ListHospitals()
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Hospitals retrieved successfully", hospitals))
}
