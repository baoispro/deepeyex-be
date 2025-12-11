package subscriptionhandler

import (
	"hospital-service/internal/config"
	"hospital-service/internal/services/subscriptionservice"
	"hospital-service/internal/utils"
	"net/http"

	"github.com/gin-gonic/gin"
)

type SubscriptionHandler struct {
	service *subscriptionservice.SubscriptionService
	cfg     config.Config
}

func NewSubscriptionHandler(cfg config.Config, service *subscriptionservice.SubscriptionService) *SubscriptionHandler {
	return &SubscriptionHandler{
		service: service,
		cfg:     cfg,
	}
}

// Subscribe - Đăng ký gói subscription
// @Summary Subscribe to a plan
// @Description User subscribes to a plan (FREE, VIP, ENTERPRISE). Old subscription will be deleted.
// @Tags Subscriptions
// @Accept json
// @Produce json
// @Param subscription body subscriptionservice.SubscribeRequest true "Subscription data"
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /subscriptions/subscribe [post]
func (h *SubscriptionHandler) Subscribe(c *gin.Context) {
	var req subscriptionservice.SubscribeRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	result, payment, err := h.service.Subscribe(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	// Nếu có payment URL, trả về để thanh toán
	if payment != nil {
		c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Payment required", payment))
		return
	}

	// Nếu là FREE, trả về subscription đã tạo
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Subscription created successfully", result))
}

// CheckAILimit - Kiểm tra có thể dùng AI không
// @Summary Check AI usage limit
// @Description Check if user can use AI diagnosis feature
// @Tags Subscriptions
// @Produce json
// @Param userId query string true "User ID"
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /subscriptions/check-ai [get]
func (h *SubscriptionHandler) CheckAILimit(c *gin.Context) {
	userID := c.Query("userId")
	if userID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "userId is required"))
		return
	}

	result, err := h.service.CheckAILimit(userID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Limit checked successfully", result))
}

// CheckConsultLimit - Kiểm tra có thể dùng tư vấn không
// @Summary Check consultation usage limit
// @Description Check if user can use consultation feature
// @Tags Subscriptions
// @Produce json
// @Param userId query string true "User ID"
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /subscriptions/check-consult [get]
func (h *SubscriptionHandler) CheckConsultLimit(c *gin.Context) {
	userID := c.Query("userId")
	if userID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "userId is required"))
		return
	}

	result, err := h.service.CheckConsultLimit(userID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Limit checked successfully", result))
}

// GetSubscription - Lấy subscription hiện tại của user
// @Summary Get current subscription
// @Description Get current active subscription of user
// @Tags Subscriptions
// @Produce json
// @Param userId query string true "User ID"
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /subscriptions [get]
func (h *SubscriptionHandler) GetSubscription(c *gin.Context) {
	userID := c.Query("userId")
	if userID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "userId is required"))
		return
	}

	result, err := h.service.GetSubscription(userID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	if result == nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, "No subscription found"))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Subscription retrieved successfully", result))
}
