package statistichandler

import (
	"hospital-service/internal/services/statisticservice"
	"hospital-service/internal/utils"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

type StatisticHandler struct {
	service *statisticservice.StatisticService
}

func NewStatisticHandler(service *statisticservice.StatisticService) *StatisticHandler {
	return &StatisticHandler{
		service: service,
	}
}

// GetStatistics - Lấy thống kê tổng quan
// @Summary Get statistics
// @Description Get statistics including orders, revenue, bookings, appointments, timeline, order status, and revenue by service
// @Tags Statistics
// @Accept json
// @Produce json
// @Param start_date query string false "Start date (YYYY-MM-DD)"
// @Param end_date query string false "End date (YYYY-MM-DD)"
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /statistics [get]
func (h *StatisticHandler) GetStatistics(c *gin.Context) {
	var startDate, endDate *time.Time

	// Parse start_date nếu có
	if startDateStr := c.Query("start_date"); startDateStr != "" {
		parsed, err := time.Parse("2006-01-02", startDateStr)
		if err != nil {
			c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Invalid start_date format. Use YYYY-MM-DD"))
			return
		}
		startDate = &parsed
	}

	// Parse end_date nếu có
	if endDateStr := c.Query("end_date"); endDateStr != "" {
		parsed, err := time.Parse("2006-01-02", endDateStr)
		if err != nil {
			c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Invalid end_date format. Use YYYY-MM-DD"))
			return
		}
		endDate = &parsed
	}

	// Validate date range
	if startDate != nil && endDate != nil && startDate.After(*endDate) {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "start_date must be before or equal to end_date"))
		return
	}

	result, err := h.service.GetStatistics(startDate, endDate)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Statistics retrieved successfully", result))
}

