package enums

type ReminderStatus string

const (
	ReminderPending ReminderStatus = "PENDING"
	ReminderDone    ReminderStatus = "DONE"
	ReminderSkipped ReminderStatus = "SKIPPED"
)
