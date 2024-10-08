from django.contrib import admin
from django.contrib.auth.hashers import make_password
from .models import User, Class
from .models import CheatingEvent

class UserAdmin(admin.ModelAdmin):
    list_display = ('nama', 'nim', 'role', 'gender', 'kelas')
    search_fields = ('nama', 'nim')

    def save_model(self, request, obj, form, change):
        # Cek jika password telah diubah
        if not obj._state.adding and obj.password != User.objects.get(pk=obj.pk).password:
            obj.password = make_password(obj.password)  # Enkripsi password
        elif obj._state.adding:  # Jika baru ditambahkan
            obj.password = make_password(obj.password)  # Enkripsi password
        super().save_model(request, obj, form, change)

@admin.register(Class)
class ClassAdmin(admin.ModelAdmin):
    list_display = ('name', 'meeting_url', 'status')
    list_editable = ('meeting_url', 'status')

@admin.register(CheatingEvent)
class CheatDetectionAdmin(admin.ModelAdmin):
    list_display = ('student_name', 'class_name', 'timestamp', 'cheating_image')

admin.site.register(User, UserAdmin)