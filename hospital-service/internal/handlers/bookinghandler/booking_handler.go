package bookinghandler

import (
	"hospital-service/internal/services/bookingservice"
	"hospital-service/internal/utils"
	"net/http"

	"github.com/gin-gonic/gin"
)

type BookingHandler struct {
	service *bookingservice.BookingService
}

func NewBookingHandler(service *bookingservice.BookingService) *BookingHandler {
	return &BookingHandler{service: service}
}

// ---------------- Create Booking ----------------
// @Summary Create a new booking
// @Description Create appointment and order in one transaction
// @Tags Bookings
// @Accept json
// @Produce json
// @Param booking body bookingservice.BookingRequest true "Booking request"
// @Success 201 {object} bookingservice.BookingResponse
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /bookings [post]
func (h *BookingHandler) CreateBooking(c *gin.Context) {
	var req bookingservice.BookingRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Invalid request: "+err.Error()))
		return
	}

	res, err := h.service.CreateBooking(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "Booking created successfully", res))
}
