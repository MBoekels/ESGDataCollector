from rest_framework import permissions

class IsAdminOrEditor(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user and (
            request.user.is_staff or 
            request.user.groups.filter(name__in=['Admin', 'Editor']).exists()
        )

class IsReadOnlyOrAdminEditor(permissions.BasePermission):
    def has_permission(self, request, view):
        if request.method in permissions.SAFE_METHODS:
            return True
        return request.user and (
            request.user.is_staff or
            request.user.groups.filter(name__in=['Admin', 'Editor']).exists()
        )
