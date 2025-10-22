package notificationrepo

import (
	"gorm.io/gorm"
	"hospital-service/internal/models/notification"
)

type NotificationRepo struct {
	db *gorm.DB
}

func NewNotificationRepo(db *gorm.DB) *NotificationRepo {
	return &NotificationRepo{db: db}
}

// Create thêm notification mới
func (r *NotificationRepo) Create(n *notification.Notification) error {
	return r.db.Create(n).Error
}

// GetAllByUserID lấy tất cả thông báo của user theo userID
func (r *NotificationRepo) GetAllByUserID(userID string) ([]notification.Notification, error) {
	var notifications []notification.Notification
	err := r.db.Where("user_id = ?", userID).Order("created_at DESC").Find(&notifications).Error
	return notifications, err
}

// MarkAsRead đánh dấu notification là đã đọc
func (r *NotificationRepo) MarkAsRead(id string) error {
	return r.db.Model(&notification.Notification{}).Where("id = ?", id).Update("read", true).Error
}

func (r *NotificationRepo) MarkAllAsReadByUserID(userID string) error {
	return r.db.Model(&notification.Notification{}).
		Where("user_id = ? AND read = ?", userID, false).
		Updates(map[string]interface{}{
			"read":       true,
			"updated_at": gorm.Expr("NOW()"),
		}).Error
}

// Delete xóa notification theo ID
func (r *NotificationRepo) Delete(id string) error {
	return r.db.Delete(&notification.Notification{}, "id = ?", id).Error
}

// DeleteAllByUserID xóa tất cả notification của user
func (r *NotificationRepo) DeleteAllByUserID(userID string) error {
	return r.db.Where("user_id = ?", userID).Delete(&notification.Notification{}).Error
}

// CountUnreadByUserID đếm số notification chưa đọc của user
func (r *NotificationRepo) CountUnreadByUserID(userID string) (int64, error) {
	var count int64
	err := r.db.Model(&notification.Notification{}).Where("user_id = ? AND read = ?", userID, false).Count(&count).Error
	return count, err
}
