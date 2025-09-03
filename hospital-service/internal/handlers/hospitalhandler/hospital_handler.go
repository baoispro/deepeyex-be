package hospitalhandler

import (
	"hospital-service/internal/config"
	"hospital-service/internal/services/hospitalservice"
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
	Name    string `json:"name" binding:"required"`
	Address string `json:"address"`
	Phone   string `json:"phone"`
	Email   string `json:"email"`
}

type updateHospitalReq struct {
	Name    string `json:"name"`
	Address string `json:"address"`
	Phone   string `json:"phone"`
	Email   string `json:"email"`
}

// ---------------- Create Hospital ----------------
// @Summary Create a new hospital
// @Description Add a new hospital record
// @Tags Hospitals
// @Accept json
// @Produce json
// @Param hospital body createHospitalReq true "Hospital info"
// @Success 201 {object} hospital.Hospital
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /hospitals [post]
func (h *HospitalHandler) CreateHospital(c *gin.Context) {
	var req createHospitalReq
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	
	created, err := h.service.CreateHospital(req.Name, req.Address, req.Phone, req.Email)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, created)
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
		c.JSON(http.StatusNotFound, gin.H{"error": "hospital not found"})
		return
	}
	c.JSON(http.StatusOK, hospitalData)
}

// ---------------- Update Hospital ----------------
// @Summary Update hospital info
// @Description Update hospital record by hospital ID
// @Tags Hospitals
// @Accept json
// @Produce json
// @Param hospital_id path string true "Hospital ID"
// @Param hospital body updateHospitalReq true "Updated hospital info"
// @Success 200 {object} hospital.Hospital
// @Failure 400 {object} map[string]string
// @Failure 404 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /hospitals/{hospital_id} [put]
func (h *HospitalHandler) UpdateHospital(c *gin.Context) {
	hospitalID := c.Param("hospital_id")
	var req updateHospitalReq

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	hospitalData, err := h.service.GetHospitalByID(hospitalID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "hospital not found"})
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

	if err := h.service.UpdateHospital(hospitalData); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, hospitalData)
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
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "hospital deleted"})
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
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, hospitals)
}
