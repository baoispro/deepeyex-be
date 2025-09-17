package orderhandler

import (
	"net/http"

	"hospital-service/internal/config"
	"hospital-service/internal/enums"
	"hospital-service/internal/models/order"
	"hospital-service/internal/services/orderservice"
	"hospital-service/internal/utils"

	"github.com/gin-gonic/gin"
)

type OrderHandler struct {
	service *orderservice.OrderService
	cfg     config.Config
}

// Request struct để update status
type UpdateOrderStatusRequest struct {
	Status enums.OrderStatus `json:"status" binding:"required"`
}

// Request struct để tạo order
type CreateOrderRequest struct {
	PatientID string                          `json:"patient_id" binding:"required"`
	Items     []orderservice.OrderItemRequest `json:"items" binding:"required"`
}

func NewOrderHandler(cfg config.Config, service *orderservice.OrderService) *OrderHandler {
	return &OrderHandler{service: service, cfg: cfg}
}

// ---------------- Create Order ----------------
// @Summary Create a new order
// @Description Create an order for a patient with list of items
// @Tags Orders
// @Accept json
// @Produce json
// @Param order body CreateOrderRequest true "Order data"
// @Success 201 {object} order.Order
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /orders [post]
func (h *OrderHandler) CreateOrder(c *gin.Context) {
	var req CreateOrderRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	if req.PatientID == "" || len(req.Items) == 0 {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "invalid order data"))
		return
	}

	// Gọi service tạo order
	o, err := h.service.CreateOrder(req.PatientID, req.Items)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "Order created successfully", o))
}

// ---------------- Get Order By ID ----------------
// @Summary Get order by ID
// @Description Retrieve an order by its ID
// @Tags Orders
// @Produce json
// @Param order_id path string true "Order ID"
// @Success 200 {object} order.Order
// @Failure 404 {object} map[string]string
// @Router /orders/{order_id} [get]
func (h *OrderHandler) GetOrderByID(c *gin.Context) {
	id := c.Param("order_id")

	o, err := h.service.GetOrder(id)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Order retrieved successfully", o))
}

// ---------------- Get Orders By Patient ----------------
// @Summary Get orders by patient ID
// @Description Retrieve all orders of a specific patient
// @Tags Orders
// @Produce json
// @Param patient_id path string true "Patient ID"
// @Success 200 {array} order.Order
// @Failure 500 {object} map[string]string
// @Router /orders/patient/{patient_id} [get]
func (h *OrderHandler) GetOrdersByPatient(c *gin.Context) {
	patientID := c.Param("patient_id")

	orders, err := h.service.GetOrdersByPatientID(patientID) // gọi service mới
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Orders retrieved successfully", orders))
}

// ---------------- Update Order Status ----------------
// @Summary Update order status
// @Description Update the status of an order (PENDING, PAID, CANCELED, DELIVERED)
// @Tags Orders
// @Accept json
// @Produce json
// @Param order_id path string true "Order ID"
// @Param status body UpdateOrderStatusRequest true "New status"
// @Success 200 {object} map[string]string
// @Failure 400 {object} map[string]string
// @Router /orders/{order_id}/status [put]
func (h *OrderHandler) UpdateOrderStatus(c *gin.Context) {
	id := c.Param("order_id")

	var req UpdateOrderStatusRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	if err := h.service.UpdateOrderStatus(id, req.Status); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Order status updated successfully", nil))
}

// ---------------- Update Order Detail ----------------
// @Summary Update order details
// @Description Update order items or other details
// @Tags Orders
// @Accept json
// @Produce json
// @Param order_id path string true "Order ID"
// @Param order body order.Order true "Updated order data"
// @Success 200 {object} map[string]string
// @Failure 400 {object} map[string]string
// @Router /orders/{order_id}/detail [put]
func (h *OrderHandler) UpdateOrderDetail(c *gin.Context) {
	id := c.Param("order_id")

	var updated order.Order
	if err := c.ShouldBindJSON(&updated); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	if err := h.service.UpdateOrderDetail(id, &updated); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Order details updated successfully", nil))
}

// ---------------- List All Orders ----------------
// @Summary List all orders
// @Description Retrieve all orders in the system
// @Tags Orders
// @Produce json
// @Success 200 {array} order.Order
// @Failure 500 {object} map[string]string
// @Router /orders [get]
func (h *OrderHandler) ListAllOrders(c *gin.Context) {
	orders, err := h.service.ListOrders()
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Orders retrieved successfully", orders))
}

// ---------------- Delete Order ----------------
// @Summary Delete an order
// @Description Remove an order by its ID
// @Tags Orders
// @Produce json
// @Param order_id path string true "Order ID"
// @Success 200 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /orders/{order_id} [delete]
func (h *OrderHandler) DeleteOrder(c *gin.Context) {
	id := c.Param("order_id")

	if err := h.service.DeleteOrder(id); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Order deleted successfully", nil))
}
