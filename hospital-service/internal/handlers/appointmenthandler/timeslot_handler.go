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

type createBatchRequest struct {
	DoctorID string `json:"doctor_id" binding:"required"`
	Slots    []struct {
		StartTime string `json:"start_time" binding:"required"` // ISO8601
		EndTime   string `json:"end_time" binding:"required"`
		Capacity  int    `json:"capacity" binding:"required"`
	} `json:"slots" binding:"required"`
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
	doctorID := c.Param("doctor_id")
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

// ---------------- Get TimeSlots By Doctor and Date ----------------
// @Summary Get time slots by doctor and specific date
// @Description Retrieve all time slots of a doctor on a given date
// @Tags TimeSlots
// @Produce json
// @Param doctor_id path string true "Doctor ID"
// @Param date query string true "Date in format YYYY-MM-DD"
// @Success 200 {array} appointment.TimeSlot
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /timeslots/doctor/{doctor_id}/date [get]
func (h *TimeSlotHandler) GetTimeSlotsByDoctorAndDate(c *gin.Context) {
	doctorID := c.Param("doctor_id")
	dateStr := c.Query("date") // YYYY-MM-DD
	if dateStr == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "date is required"))
		return
	}

	date, err := time.Parse("2006-01-02", dateStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "invalid date format, expected YYYY-MM-DD"))
		return
	}

	slots, err := h.service.GetByDoctorAndDate(doctorID, date)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "time slots retrieved successfully", slots))
}

// ---------------- Get TimeSlots By Doctor and Month ----------------
// @Summary Get time slots by doctor and month
// @Description Retrieve all time slots of a doctor in a given month
// @Tags TimeSlots
// @Produce json
// @Param doctor_id path string true "Doctor ID"
// @Param month query string true "Month in format YYYY-MM (e.g., 2025-09)"
// @Success 200 {array} appointment.TimeSlot
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /timeslots/doctor/{doctor_id}/month [get]
func (h *TimeSlotHandler) GetTimeSlotsByDoctorAndMonth(c *gin.Context) {
	doctorID := c.Param("doctor_id")
	monthStr := c.Query("month") // YYYY-MM
	if monthStr == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "month is required"))
		return
	}

	date, err := time.Parse("2006-01", monthStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "invalid month format, expected YYYY-MM"))
		return
	}

	slots, err := h.service.GetByDoctorAndMonth(doctorID, date)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "time slots retrieved successfully", slots))
}

// ---------------- Get TimeSlots By Doctor and Date Range ----------------
// @Summary Get time slots by doctor and date range
// @Description Retrieve all time slots of a doctor in a given date range
// @Tags TimeSlots
// @Produce json
// @Param doctor_id path string true "Doctor ID"
// @Param start_date query string true "Start date in format YYYY-MM-DD"
// @Param end_date query string true "End date in format YYYY-MM-DD"
// @Success 200 {array} appointment.TimeSlot
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /timeslots/doctor/{doctor_id}/date-range [get]
func (h *TimeSlotHandler) GetTimeSlotsByDoctorAndDateRange(c *gin.Context) {
	doctorID := c.Param("doctor_id")
	startDateStr := c.Query("start_date")
	endDateStr := c.Query("end_date")
	if startDateStr == "" || endDateStr == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "start_date and end_date are required"))
		return
	}
	
	startDate, err := time.Parse("2006-01-02", startDateStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "invalid start date format, expected YYYY-MM-DD"))
		return
	}
	
	endDate, err := time.Parse("2006-01-02", endDateStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "invalid end date format, expected YYYY-MM-DD"))
		return
	}
	
	slots, err := h.service.GetByDoctorAndDateRange(doctorID, startDate, endDate)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}
	
	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "time slots retrieved successfully", slots))
}


// ---------------- Create Batch TimeSlots ----------------
// @Summary Create batch time slots
// @Description Create multiple time slots in a single request
// @Tags TimeSlots
// @Accept json
// @Produce json
// @Param request body createBatchRequest true "Batch create time slots request"
// @Success 201 {object} map[string]interface{} "{"message":"time slots created successfully","data":[...]}"
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /timeslots/batch [post]
func (h *TimeSlotHandler) CreateBatch(c *gin.Context) {
	var req createBatchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	var inputs []appointmentservice.TimeSlotInput
	for _, s := range req.Slots {
		start, err1 := time.Parse(time.RFC3339, s.StartTime)
		end, err2 := time.Parse(time.RFC3339, s.EndTime)
		if err1 != nil || err2 != nil {
			c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "invalid time format, must be RFC3339"))
			return
		}
		inputs = append(inputs, appointmentservice.TimeSlotInput{
			StartTime: start,
			EndTime:   end,
			Capacity:  s.Capacity,
		})
	}

	slots, err := h.service.CreateBatch(req.DoctorID, inputs)
	if err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "Time slots created successfully", slots))
}

// @Summary Create multiple shift slots
// @Description Create time slots for doctor by providing list of dates and shifts
// @Tags TimeSlots
// @Accept json
// @Produce json
// @Param request body appointmentservice.CreateMultiShiftSlotsRequest true "Multi shift slot request"
// @Success 201 {object} map[string]interface{}
// @Failure 400 {object} map[string]string
// @Router /timeslots/multi-shift [post]
func (h *TimeSlotHandler) CreateMultiShift(c *gin.Context) {
    var req appointmentservice.CreateMultiShiftSlotsRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
        return
    }

    slots, err := h.service.CreateMultiShiftSlots(req)
    if err != nil {
        c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
        return
    }

    c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "Time slots created successfully", slots))
}

// @Summary Import doctor day-off from Excel
// @Description Upload excel file to delete doctor timeslots for off-days
// @Tags TimeSlots
// @Accept multipart/form-data
// @Produce json
// @Param file formData file true "Excel file"
// @Success 200 {object} map[string]string
// @Failure 400 {object} map[string]string
// @Router /timeslots/import-dayoff [post]
func (h *TimeSlotHandler) ImportDoctorDayOff(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "file is required"))
		return
	}

	filePath := "./uploads/" + file.Filename
	if err := c.SaveUploadedFile(file, filePath); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, "failed to save file"))
		return
	}

	if err := h.service.ImportDoctorDayOff(filePath); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "import and delete day-off slots successfully", nil))
}


