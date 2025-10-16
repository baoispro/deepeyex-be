package fullrecordhandler

import (
	"hospital-service/internal/services/fullrecordservice"
	"hospital-service/internal/utils"
	"net/http"

	"github.com/gin-gonic/gin"
)

type FullRecordHandler struct {
	service *fullrecordservice.FullRecordService
}

func NewFullRecordHandler(service *fullrecordservice.FullRecordService) *FullRecordHandler {
	return &FullRecordHandler{service: service}
}

// ---------------- Create Full Record ----------------
// @Summary Create full medical record
// @Description Create a new medical record with attachments and prescription
// @Tags Medical Records
// @Accept multipart/form-data
// @Produce json
// @Param data body fullrecordservice.FullRecordCreateRequest true "Full record data"
// @Success 201 {object} medicalrecord.MedicalRecord
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /full-records/full [post]
func (h *FullRecordHandler) CreateFullRecord(c *gin.Context) {
	var req fullrecordservice.FullRecordCreateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Invalid request: "+err.Error()))
		return
	}

	record, err := h.service.CreateFullRecord(&req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "Full medical record created successfully", record))
}

// ---------------- Complete Record ----------------
// @Summary Complete existing medical record
// @Description Update an existing record with diagnosis, notes, attachments, and prescription
// @Tags Medical Records
// @Accept json
// @Produce json
// @Param record body fullrecordservice.CompleteRecordRequest true "Complete record data"
// @Success 200 {object} medicalrecord.MedicalRecord
// @Failure 400 {object} map[string]string
// @Failure 404 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /full-records/complete [put]
func (h *FullRecordHandler) CompleteRecord(c *gin.Context) {
	var req fullrecordservice.CompleteRecordRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Invalid request: "+err.Error()))
		return
	}

	record, err := h.service.CompleteRecord(&req)
	if err != nil {
		// kiểm tra lỗi not found hoặc khác
		if err.Error() == "medical record not found" {
			c.JSON(http.StatusNotFound, utils.ErrorResponse(http.StatusNotFound, err.Error()))
		} else {
			c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		}
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Medical record completed successfully", record))
}
