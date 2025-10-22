package notificationservice

import (
	"hospital-service/internal/models/notification"
	"hospital-service/internal/repositories/notificationrepo"
	"hospital-service/internal/websocket"
	"time"

	"github.com/google/uuid"
)

type NotificationService struct {
	repo *notificationrepo.NotificationRepo
	wsHub *websocket.Hub // thêm hub để broadcast
}

func NewNotificationService(repo *notificationrepo.NotificationRepo, wsHub *websocket.Hub) *NotificationService {
	return &NotificationService{repo: repo, wsHub: wsHub,}
}

// CreateNotification tạo thông báo mới
func (s *NotificationService) CreateNotification(userID, title, message, targetURL string) (*notification.Notification, error) {
	noti := &notification.Notification{
		ID:        uuid.NewString(),
		UserID:    userID,
		Title:     title,
		Message:   message,
		TargetURL: targetURL,
		Read:      false,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	err := s.repo.Create(noti)

	if s.wsHub != nil {
		payload := map[string]interface{}{
			"notification": noti,
			"message":      message,
		}
		go s.wsHub.BroadcastToPatient(userID, websocket.NewNotification, payload)
	}
	return noti, err
}

// GetAllNotifications lấy tất cả thông báo của user
func (s *NotificationService) GetAllNotifications(userID string) ([]notification.Notification, error) {
	return s.repo.GetAllByUserID(userID)
}

// MarkNotificationRead đánh dấu notification là đã đọc
func (s *NotificationService) MarkNotificationRead(id string) error {
	return s.repo.MarkAsRead(id)
}

func (s *NotificationService) MarkAllNotificationsRead(userID string) error {
    return s.repo.MarkAllAsReadByUserID(userID)
}

// DeleteNotification xóa notification theo ID
func (s *NotificationService) DeleteNotification(id string) error {
	return s.repo.Delete(id)
}

// DeleteAllNotifications xóa tất cả thông báo của user
func (s *NotificationService) DeleteAllNotifications(userID string) error {
	return s.repo.DeleteAllByUserID(userID)
}

// CountUnreadNotifications đếm số thông báo chưa đọc của user
func (s *NotificationService) CountUnreadNotifications(userID string) (int64, error) {
	return s.repo.CountUnreadByUserID(userID)
}
