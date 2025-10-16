package servicehandler

import (
	"hospital-service/internal/config"
	"hospital-service/internal/services/doctorserviceservice"
	"hospital-service/internal/utils"
	"net/http"

	"github.com/gin-gonic/gin"
)

type ServiceHandler struct {
	service *doctorserviceservice.ServiceService
	cfg     config.Config
}

// Request structs
type CreateServiceRequest struct {
	Name     string  `json:"name" binding:"required"`
	Duration int     `json:"duration" binding:"required"`
	Price    float64 `json:"price" binding:"required"`
}

type UpdateServiceRequest struct {
	Name     string  `json:"name"`
	Duration int     `json:"duration"`
	Price    float64 `json:"price"`
}

type AssignServiceRequest struct {
	DoctorID  string `json:"doctor_id" binding:"required"`
	ServiceID string `json:"service_id" binding:"required"`
}

// NewServiceHandler - khởi tạo handler
func NewServiceHandler(cfg config.Config, service *doctorserviceservice.ServiceService) *ServiceHandler {
	return &ServiceHandler{service: service, cfg: cfg}
}

// ============== SERVICE CRUD ==============

// CreateService - Tạo service mới
// @Summary Create a new service
// @Description Admin creates a new service
// @Tags Services
// @Accept json
// @Produce json
// @Param service body CreateServiceRequest true "Service data"
// @Success 201 {object} map[string]interface{}
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /services [post]
func (h *ServiceHandler) CreateService(c *gin.Context) {
	var req CreateServiceRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	service, err := h.service.CreateService(req.Name, req.Duration, req.Price)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "Service created successfully", service))
}

// GetServiceByID - Lấy service theo ID
// @Summary Get service by ID
// @Description Get a specific service by ID
// @Tags Services
// @Produce json
// @Param service_id path string true "Service ID"
// @Success 200 {object} map[string]interface{}
// @Failure 404 {object} map[string]string
// @Router /services/{service_id} [get]
func (h *ServiceHandler) GetServiceByID(c *gin.Context) {
	serviceID := c.Param("service_id")

	service, err := h.service.GetServiceByID(serviceID)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, "Service not found"))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Service retrieved successfully", service))
}

// ListAllServices - Lấy tất cả services
// @Summary List all services
// @Description Get all available services with optional filters
// @Tags Services
// @Produce json
// @Param name query string false "Filter by service name (partial match)"
// @Param duration query int false "Filter by duration in minutes (exact match)"
// @Success 200 {object} map[string]interface{}
// @Failure 500 {object} map[string]string
// @Router /services [get]
func (h *ServiceHandler) ListAllServices(c *gin.Context) {
	// Lấy query params
	name := c.Query("name")
	duration := c.Query("duration")

	services, err := h.service.ListAllServices(name, duration)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Services retrieved successfully", services))
}

// UpdateService - Cập nhật service
// @Summary Update a service
// @Description Admin updates service information
// @Tags Services
// @Accept json
// @Produce json
// @Param service_id path string true "Service ID"
// @Param service body UpdateServiceRequest true "Updated service data"
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]string
// @Failure 404 {object} map[string]string
// @Router /services/{service_id} [put]
func (h *ServiceHandler) UpdateService(c *gin.Context) {
	serviceID := c.Param("service_id")

	var req UpdateServiceRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	service, err := h.service.UpdateService(serviceID, req.Name, req.Duration, req.Price)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Service updated successfully", service))
}

// DeleteService - Xóa service
// @Summary Delete a service
// @Description Admin deletes a service
// @Tags Services
// @Produce json
// @Param service_id path string true "Service ID"
// @Success 200 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /services/{service_id} [delete]
func (h *ServiceHandler) DeleteService(c *gin.Context) {
	serviceID := c.Param("service_id")

	err := h.service.DeleteService(serviceID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Service deleted successfully", nil))
}

// ============== DOCTOR-SERVICE RELATIONSHIP ==============

// ListServicesByDoctorID - Lấy services theo doctor ID
// @Summary List services by doctor ID
// @Description Retrieve all services for a specific doctor
// @Tags Services
// @Produce json
// @Param doctor_id path string true "Doctor ID"
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /doctors/{doctor_id}/services [get]
func (h *ServiceHandler) ListServicesByDoctorID(c *gin.Context) {
	doctorID := c.Param("doctor_id")

	if doctorID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "doctor_id is required"))
		return
	}

	services, err := h.service.GetServicesByDoctorID(doctorID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Services retrieved successfully", services))
}

// AssignServiceToDoctor - Gán service cho bác sĩ
// @Summary Assign service to doctor
// @Description Admin assigns a service to a doctor
// @Tags Services
// @Accept json
// @Produce json
// @Param data body AssignServiceRequest true "Doctor and Service IDs"
// @Success 200 {object} map[string]string
// @Failure 400 {object} map[string]string
// @Router /services/assign [post]
func (h *ServiceHandler) AssignServiceToDoctor(c *gin.Context) {
	var req AssignServiceRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	err := h.service.AssignServiceToDoctor(req.DoctorID, req.ServiceID)
	if err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Service assigned to doctor successfully", nil))
}

// RemoveServiceFromDoctor - Xóa service khỏi bác sĩ
// @Summary Remove service from doctor
// @Description Admin removes a service from a doctor
// @Tags Services
// @Produce json
// @Param doctor_id path string true "Doctor ID"
// @Param service_id path string true "Service ID"
// @Success 200 {object} map[string]string
// @Failure 400 {object} map[string]string
// @Router /doctors/{doctor_id}/services/{service_id} [delete]
func (h *ServiceHandler) RemoveServiceFromDoctor(c *gin.Context) {
	doctorID := c.Param("doctor_id")
	serviceID := c.Param("service_id")

	if doctorID == "" || serviceID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "doctor_id and service_id are required"))
		return
	}

	err := h.service.RemoveServiceFromDoctor(doctorID, serviceID)
	if err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Service removed from doctor successfully", nil))
}
