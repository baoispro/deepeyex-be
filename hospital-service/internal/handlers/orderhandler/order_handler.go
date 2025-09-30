package orderhandler

import (
	"net/http"

	"hospital-service/internal/config"
	"hospital-service/internal/enums"
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

type UpdateOrderAppointmentRequest struct {
	AppointmentID string `json:"appointment_id" binding:"required"`
}

func NewOrderHandler(cfg config.Config, service *orderservice.OrderService) *OrderHandler {
	return &OrderHandler{service: service, cfg: cfg}
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

// ---------------- Update Order Appointment ----------------
// @Summary Update order appointment
// @Description Update the appointment ID of an order (change linked appointment)
// @Tags Orders
// @Accept json
// @Produce json
// @Param order_id path string true "Order ID"
// @Param appointment body UpdateOrderAppointmentRequest true "New appointment ID"
// @Success 200 {object} map[string]string
// @Failure 400 {object} map[string]string
// @Failure 404 {object} map[string]string
// @Router /orders/{order_id}/appointment [put]
func (h *OrderHandler) UpdateOrderAppointment(c *gin.Context) {
	id := c.Param("order_id")

	var req UpdateOrderAppointmentRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	if req.AppointmentID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "appointment_id is required"))
		return
	}

	if err := h.service.UpdateOrderAppointment(id, req.AppointmentID); err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Order appointment updated successfully", nil))
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
