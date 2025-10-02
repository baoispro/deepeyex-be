package paymenthandler

import (
	"hospital-service/internal/services/paymentservice"
	"hospital-service/internal/utils"
	"net/http"

	"github.com/gin-gonic/gin"
)

type VnpayHandler struct {
	service *paymentservice.VnpayService
}

func NewVnpayHandler(service *paymentservice.VnpayService) *VnpayHandler {
	return &VnpayHandler{service: service}
}

// ---------------- Create Payment ----------------
// @Summary Create VNPAY payment
// @Tags Payments
// @Accept json
// @Produce json
// @Param payment body map[string]interface{} true "Payment request"
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
// @Param query query map[string]string true "Return query"
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
