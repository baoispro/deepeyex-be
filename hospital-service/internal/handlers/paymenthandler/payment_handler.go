package paymenthandler

import (
	"hospital-service/internal/services/paymentservice"
	"hospital-service/internal/services/subscriptionservice"
	"hospital-service/internal/utils"
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
)

type VnpayHandler struct {
	service            *paymentservice.VnpayService
	subscriptionService *subscriptionservice.SubscriptionService
}

func NewVnpayHandler(service *paymentservice.VnpayService, subscriptionService *subscriptionservice.SubscriptionService) *VnpayHandler {
	return &VnpayHandler{
		service:            service,
		subscriptionService: subscriptionService,
	}
}

// ---------------- Create Payment ----------------
// @Summary Create VNPAY payment
// @Tags Payments
// @Accept json
// @Produce json
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /vnpay/create-payment [post]
func (h *VnpayHandler) CreatePayment(c *gin.Context) {
	var req struct {
		Amount  int    `json:"amount"`
		OrderID string `json:"orderId"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Invalid request: "+err.Error()))
		return
	}

	paymentUrl, err := h.service.CreatePaymentURL(req.Amount, req.OrderID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, "Failed to create payment URL: "+err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Payment URL created", map[string]interface{}{
		"paymentUrl": paymentUrl,
	}))
}

// ---------------- VNPAY Return ----------------
// @Summary Handle VNPAY return URL
// @Tags Payments
// @Produce json
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]string
// @Router /vnpay/return [get]
func (h *VnpayHandler) VnpayReturn(c *gin.Context) {
	query := c.Request.URL.Query()

	if !h.service.VerifyReturn(query) {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Dữ liệu không hợp lệ"))
		return
	}

	statusCode := query.Get("vnp_ResponseCode")
	orderId := query.Get("vnp_TxnRef")
	paymentType := query.Get("type")

	// Nếu là subscription payment
	if paymentType == "subscription" {
		if statusCode == "00" {
			subscriptionID := query.Get("subscriptionId")
			userID := query.Get("userId")
			planName := query.Get("planName")
			durationStr := query.Get("duration")

			duration, err := strconv.Atoi(durationStr)
			if err != nil {
				duration = 30 // default
			}

			// Tạo subscription sau khi thanh toán thành công
			result, err := h.subscriptionService.CompleteSubscription(subscriptionID, userID, planName, duration)
			if err != nil {
				c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, "Failed to create subscription: "+err.Error()))
				return
			}

			c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Thanh toán và đăng ký gói thành công", map[string]interface{}{
				"subscriptionId": subscriptionID,
				"status":         "success",
				"subscription":   result,
			}))
		} else {
			c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Thanh toán thất bại, mã lỗi: "+statusCode, map[string]interface{}{
				"subscriptionId": orderId,
				"status":         "failed",
				"code":           statusCode,
			}))
		}
		return
	}

	// Xử lý order payment bình thường
	if statusCode == "00" {
		c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Thanh toán thành công", map[string]interface{}{
			"orderId": orderId,
			"status":  "success",
		}))
	} else {
		c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Thanh toán thất bại, mã lỗi: "+statusCode, map[string]interface{}{
			"orderId": orderId,
			"status":  "failed",
			"code":    statusCode,
		}))
	}
}
