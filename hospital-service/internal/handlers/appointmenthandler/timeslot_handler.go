package appointmenthandler

import (
	"net/http"
	"time"

	"hospital-service/internal/config"
	"hospital-service/internal/services/appointmentservice"
	"hospital-service/internal/utils"

	"github.com/gin-gonic/gin"
)

type TimeSlotHandler struct {
	service *appointmentservice.TimeSlotService
	cfg     config.Config
}

// Request struct chỉ chứa field client cần gửi
type createTimeSlotReq struct {
    DoctorID  string    `json:"doctor_id" binding:"required"`
    StartTime time.Time `json:"start_time" binding:"required"`
    EndTime   time.Time `json:"end_time" binding:"required"`
    Capacity  int       `json:"capacity" binding:"required"`
}

type updateTimeSlotReq struct {
	StartTime *time.Time `json:"start_time,omitempty"`
	EndTime   *time.Time `json:"end_time,omitempty"`
	Capacity  *int       `json:"capacity,omitempty"`
}

func NewTimeSlotHandler(cfg config.Config, service *appointmentservice.TimeSlotService) *TimeSlotHandler {
	return &TimeSlotHandler{service: service, cfg: cfg}
}

// ---------------- Create TimeSlot ----------------
// @Summary Create a new time slot
// @Description Add a new time slot for a doctor
// @Tags TimeSlots
// @Accept json
// @Produce json
// @Param timeslot body createTimeSlotReq true "TimeSlot data"
// @Success 201 {object} appointment.TimeSlot
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /timeslots [post]
func (h *TimeSlotHandler) CreateTimeSlot(c *gin.Context) {

	var req createTimeSlotReq
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	// Gọi service Create, service sẽ tự set SlotID, CreatedAt, UpdatedAt
	slot, err := h.service.Create(req.DoctorID, req.StartTime, req.EndTime, req.Capacity)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "Time slot created successfully", slot))
}



// ---------------- Get TimeSlot By ID ----------------
// @Summary Get time slot by ID
// @Description Retrieve time slot info by slot_id
// @Tags TimeSlots
// @Produce json
// @Param slot_id path string true "Slot ID"
// @Success 200 {object} appointment.TimeSlot
// @Failure 404 {object} map[string]string
// @Router /timeslots/{slot_id} [get]
func (h *TimeSlotHandler) GetTimeSlotByID(c *gin.Context) {
	id := c.Param("slot_id")
	
	slot, err := h.service.GetByID(id)
	if err != nil {
		c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, "time slot not found"))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "time slot retrieved successfully", slot))
}

// ---------------- Get TimeSlots By Doctor ----------------
// @Summary Get time slots by doctor ID
// @Description List all time slots of a doctor
// @Tags TimeSlots
// @Produce json
// @Param doctor_id path string true "Doctor ID"
// @Success 200 {array} appointment.TimeSlot
// @Failure 500 {object} map[string]string
// @Router /timeslots/doctor/{doctor_id} [get]
func (h *TimeSlotHandler) GetTimeSlotsByDoctor(c *gin.Context) {
	doctorID:= c.Param("doctor_id")
	slots, err := h.service.GetByDoctorID(doctorID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "time slots retrieved successfully", slots))
}

// ---------------- Update TimeSlot ----------------
// @Summary Update time slot
// @Description Update existing time slot (start_time, end_time, capacity)
// @Tags TimeSlots
// @Accept json
// @Produce json
// @Param slot_id path string true "Slot ID"
// @Param timeslot body updateTimeSlotReq true "Updated TimeSlot data"
// @Success 200 {object} appointment.TimeSlot
// @Failure 400 {object} map[string]string
// @Router /timeslots/{slot_id} [put]
func (h *TimeSlotHandler) UpdateTimeSlot(c *gin.Context) {
	slotID := c.Param("slot_id")
	var req updateTimeSlotReq
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	// Gọi service Update
	updatedSlot, err := h.service.Update(slotID, req.StartTime, req.EndTime, req.Capacity)
	if err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Time slot updated successfully", updatedSlot))
}

// ---------------- Delete TimeSlot ----------------
// @Summary Delete time slot
// @Description Delete a time slot by slot_id
// @Tags TimeSlots
// @Produce json
// @Param slot_id path string true "Slot ID"
// @Success 200 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /timeslots/{slot_id} [delete]
func (h *TimeSlotHandler) DeleteTimeSlot(c *gin.Context) {
	id := c.Param("slot_id")
	if err := h.service.Delete(id); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "time slot deleted successfully", nil))
}

// ---------------- List All TimeSlots ----------------
// @Summary List all time slots
// @Description Retrieve all time slots
// @Tags TimeSlots
// @Produce json
// @Success 200 {array} appointment.TimeSlot
// @Failure 500 {object} map[string]string
// @Router /timeslots [get]
func (h *TimeSlotHandler) ListAllTimeSlots(c *gin.Context) {
	slots, err := h.service.ListAll()
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "time slots retrieved successfully", slots))
}
