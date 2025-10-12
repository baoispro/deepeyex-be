package uploadhandler

import (
	"hospital-service/internal/services/uploadservice"
	"hospital-service/internal/utils"
	"net/http"

	"github.com/gin-gonic/gin"
)

type UploadHandler struct {
	service *uploadservice.UploadService
}

func NewUploadHandler(service *uploadservice.UploadService) *UploadHandler {
	return &UploadHandler{service: service}
}

// @Summary Upload a file
// @Description Upload file to S3 and return its URL
// @Tags Upload
// @Accept multipart/form-data
// @Produce json
// @Param file formData file true "File to upload"
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /upload [post]
func (h *UploadHandler) UploadFile(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "file is required"))
		return
	}

	url, err := h.service.UploadFile(file)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "File uploaded successfully", gin.H{
		"url": url,
	}))
}
