package medicalrecordhandler

import (
	"net/http"

	"hospital-service/internal/config"
	"hospital-service/internal/models/medicalrecord"
	"hospital-service/internal/services/medicalrecordservice"
	"hospital-service/internal/utils"

	"github.com/gin-gonic/gin"
)

type AttachmentHandler struct {
	service *medicalrecordservice.AttachmentService
	cfg     config.Config
}

func NewAttachmentHandler(cfg config.Config, s *medicalrecordservice.AttachmentService) *AttachmentHandler {
	return &AttachmentHandler{service: s, cfg: cfg}
}

// AddAttachment godoc
// @Summary Thêm file đính kèm vào medical record
// @Description Upload attachment cho một medical record
// @Tags Attachments
// @Accept multipart/form-data
// @Produce json
// @Param record_id formData string true "Record ID"
// @Param file formData file true "File để upload"
// @Param file_type formData string true "File type (image/pdf/...)"
// @Success 201 {object} medicalrecord.Attachment
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /attachments [post]
func (h *AttachmentHandler) AddAttachment(c *gin.Context) {
	recordID := c.PostForm("record_id")
	fileType := c.PostForm("file_type")

	if recordID == "" || fileType == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "record_id and file_type are required"))
		return
	}

	// Lấy file upload
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "file is required"))
		return
	}

	att := &medicalrecord.Attachment{
		RecordID: recordID,
		FileType: fileType,
	}

	// Gọi service upload lên S3 và lưu DB
	savedAtt, err := h.service.AddAttachment(att, file)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "Attachment added successfully", savedAtt))
}

// GetAttachments godoc
// @Summary Lấy danh sách file đính kèm của một record
// @Description Trả về danh sách các attachment theo record_id
// @Tags Attachments
// @Accept json
// @Produce json
// @Param record_id path string true "Record ID"
// @Success 200 {array} medicalrecord.Attachment
// @Failure 400 {object} map[string]string
// @Router /attachments/{record_id}/medical_records [get]
func (h *AttachmentHandler) GetAttachments(c *gin.Context) {
	recordID := c.Param("record_id")
	if recordID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "record_id is required"))
		return
	}

	attachments, err := h.service.GetAttachmentsByRecordID(recordID)
	if err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Attachments retrieved successfully", attachments))
}

// DeleteAttachment godoc
// @Summary Xóa một file đính kèm
// @Description Xóa attachment theo ID
// @Tags Attachments
// @Accept json
// @Produce json
// @Param id path string true "Attachment ID"
// @Success 200 {object} map[string]string
// @Failure 400 {object} map[string]string
// @Router /attachments/{id} [delete]
func (h *AttachmentHandler) DeleteAttachment(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "attachment id is required"))
		return
	}

	if err := h.service.DeleteAttachmentByID(id); err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Attachment deleted successfully", nil))
}
