package medicalrecordhandler

import (
	"net/http"
	"time"

	"hospital-service/internal/config"
	"hospital-service/internal/services/medicalrecordservice"
	"hospital-service/internal/utils"

	"github.com/gin-gonic/gin"
)

type FollowUpHandler struct {
	service *medicalrecordservice.FollowUpService
	cfg     config.Config

}
type UpdateFollowUpRequest struct {
    Notes           string     `json:"notes"`
    NextAppointment *time.Time `json:"next_appointment"`
}

func NewFollowUpHandler(cfg config.Config ,s *medicalrecordservice.FollowUpService) *FollowUpHandler {
	return &FollowUpHandler{service: s, cfg: cfg}
}

// ---------------- POST ----------------
// @Summary Tạo lịch tái khám
// @Description Thêm follow-up cho medical record
// @Tags FollowUps
// @Accept json
// @Produce json
// @Param record_id path string true "Medical Record ID"
// @Param data body UpdateFollowUpRequest true "FollowUp Data"
// @Success 201 {object} map[string]interface{}
// @Failure 400 {object} map[string]string
// @Router /followups/{record_id}/medical_records [post]
func (h *FollowUpHandler) CreateFollowUp(c *gin.Context) {
    recordID := c.Param("record_id")
    if recordID == "" {
        c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "record_id is required"))
        return
    }

    var req UpdateFollowUpRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
        return
    }

    followUp, err := h.service.CreateFollowUp(recordID, req.Notes, "", req.NextAppointment)
    if err != nil {
        c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
        return
    }

    c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "FollowUp created successfully", followUp))
}

// ---------------- GET  ----------------
// @Summary Lấy danh sách lịch tái khám
// @Description Trả về tất cả follow-up của medical record
// @Tags FollowUps
// @Accept json
// @Produce json
// @Param record_id path string true "Medical Record ID"
// @Success 200 {array} map[string]interface{}
// @Failure 400 {object} map[string]string
// @Router /followups/{record_id}/medical_records [get]
func (h *FollowUpHandler) GetFollowUps(c *gin.Context) {
    recordID := c.Param("record_id")
    if recordID == "" {
        c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "record_id is required"))
        return
    }

    followUps, err := h.service.GetFollowUpsByRecordID(recordID)
    if err != nil {
        c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
        return
    }

    c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "FollowUps retrieved successfully", followUps))
}

// ---------------- PUT /followups/{id} ----------------
// @Summary Cập nhật lịch tái khám
// @Description Cập nhật notes hoặc next appointment
// @Tags FollowUps
// @Accept json
// @Produce json
// @Param follow_up_id path string true "FollowUp ID"
// @Param data body UpdateFollowUpRequest true "Update Data"
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]string
// @Router /followups/{follow_up_id} [put]
func (h *FollowUpHandler) UpdateFollowUp(c *gin.Context) {
    followUpID := c.Param("follow_up_id")
    if followUpID == "" {
        c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "follow_up_id is required"))
        return
    }

    var req UpdateFollowUpRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
        return
    }

    updated, err := h.service.UpdateFollowUp(followUpID, req.Notes, req.NextAppointment)
    if err != nil {
        c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
        return
    }

    c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "FollowUp updated successfully", updated))
}

// ---------------- DELETE /followups/{id} ----------------
// @Summary Xóa lịch tái khám
// @Description Xóa follow-up theo ID
// @Tags FollowUps
// @Accept json
// @Produce json
// @Param follow_up_id path string true "FollowUp ID"
// @Success 200 {object} map[string]string
// @Failure 400 {object} map[string]string
// @Router /followups/{follow_up_id} [delete]
func (h *FollowUpHandler) DeleteFollowUp(c *gin.Context) {
    followUpID := c.Param("follow_up_id")
    if followUpID == "" {
        c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "follow_up_id is required"))
        return
    }

    if err := h.service.DeleteFollowUp(followUpID); err != nil {
        c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
        return
    }

    c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "FollowUp deleted successfully", nil))
}