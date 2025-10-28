package notificationhandler

import (
	"hospital-service/internal/services/notificationservice"
	"hospital-service/internal/utils"
	"net/http"

	"github.com/gin-gonic/gin"
)

type NotificationHandler struct {
	service *notificationservice.NotificationService
}

func NewNotificationHandler(service *notificationservice.NotificationService) *NotificationHandler {
	return &NotificationHandler{service: service}
}

// ---------------- Create Notification ----------------
// @Summary Create a new notification
// @Description Create a notification for a user
// @Tags Notifications
// @Accept json
// @Produce json
// @Param userID query string true "User ID"
// @Param title query string true "Title"
// @Param message query string true "Message"
// @Param targetURL query string false "Target URL"
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /notifications [post]
func (h *NotificationHandler) CreateNotification(c *gin.Context) {
	userID := c.Query("userID")
	title := c.Query("title")
	message := c.Query("message")
	targetURL := c.Query("targetURL")

	if userID == "" || title == "" || message == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Missing required fields"))
		return
	}

	noti, err := h.service.CreateNotification(userID, title, message, targetURL)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusCreated, utils.SuccessResponse(http.StatusCreated, "Notification created", noti))
}

// ---------------- Get All Notifications ----------------
// @Summary Get all notifications of a user
// @Description List all notifications for a user
// @Tags Notifications
// @Produce json
// @Param userID query string true "User ID"
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /notifications [get]
func (h *NotificationHandler) GetAllNotifications(c *gin.Context) {
	userID := c.Query("userID")
	if userID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Missing userID"))
		return
	}

	notis, err := h.service.GetAllNotifications(userID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Notifications fetched", notis))
}

// ---------------- Mark Notification Read ----------------
// @Summary Mark a notification as read
// @Description Mark notification as read by ID
// @Tags Notifications
// @Param id path string true "Notification ID"
// @Success 200 {object} map[string]string
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /notifications/{id}/read [put]
func (h *NotificationHandler) MarkNotificationRead(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Missing notification ID"))
		return
	}

	if err := h.service.MarkNotificationRead(id); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Notification marked as read", nil))
}

// ---------------- Mark All Notifications Read ----------------
// @Summary Mark all notifications as read
// @Description Mark all notifications of a user as read
// @Tags Notifications
// @Param userId path string true "User ID"
// @Success 200 {object} map[string]string
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /notifications/user/{userId}/read-all [put]
func (h *NotificationHandler) MarkAllNotificationsRead(c *gin.Context) {
	userID := c.Param("userId")
	if userID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Missing user ID"))
		return
	}

	if err := h.service.MarkAllNotificationsRead(userID); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "All notifications marked as read", nil))
}

// ---------------- Delete Notification ----------------
// @Summary Delete a notification by ID
// @Description Delete notification
// @Tags Notifications
// @Param id path string true "Notification ID"
// @Success 200 {object} map[string]string
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /notifications/{id} [delete]
func (h *NotificationHandler) DeleteNotification(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Missing notification ID"))
		return
	}

	if err := h.service.DeleteNotification(id); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Notification deleted", nil))
}

// ---------------- Delete All Notifications ----------------
// @Summary Delete all notifications of a user
// @Description Delete all notifications for a user
// @Tags Notifications
// @Param userID query string true "User ID"
// @Success 200 {object} map[string]string
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /notifications/all [delete]
func (h *NotificationHandler) DeleteAllNotifications(c *gin.Context) {
	userID := c.Query("userID")
	if userID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Missing userID"))
		return
	}

	if err := h.service.DeleteAllNotifications(userID); err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "All notifications deleted", nil))
}

// ---------------- Count Unread Notifications ----------------
// @Summary Count unread notifications
// @Description Count number of unread notifications for a user
// @Tags Notifications
// @Param userID query string true "User ID"
// @Success 200 {object} map[string]int64
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /notifications/unread [get]
func (h *NotificationHandler) CountUnreadNotifications(c *gin.Context) {
	userID := c.Query("userID")
	if userID == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Missing userID"))
		return
	}

	count, err := h.service.CountUnreadNotifications(userID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Unread notifications counted", map[string]int64{"unread": count}))
}
