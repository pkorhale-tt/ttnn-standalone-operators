import inspect
import ttnn

print("ttnn.device attributes:\n")
for name in dir(ttnn.device):
    print("  ", name)

print("\nDispatchCoreConfig type / constructor:")
DispatchCoreConfig = getattr(ttnn.device, "DispatchCoreConfig", None)
print("DispatchCoreConfig:", DispatchCoreConfig)

if DispatchCoreConfig is not None:
    print("\nDispatchCoreConfig signature / help:")
    try:
        print(inspect.getsource(DispatchCoreConfig))
    except OSError:
        print("  (source not available, try help() instead)")
        help(DispatchCoreConfig)
