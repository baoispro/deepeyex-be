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

// NewServiceHandler - khởi tạo handler
func NewServiceHandler(cfg config.Config, service *doctorserviceservice.ServiceService) *ServiceHandler {
	return &ServiceHandler{service: service, cfg: cfg}
}

// ---------------- List Services by Doctor ID ----------------
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
